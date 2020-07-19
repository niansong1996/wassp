# -*- coding: utf-8 -*-
import json
import os
import codecs
import pickle
import babel
import unicodedata
import re
from tqdm import tqdm
from babel.numbers import parse_decimal, NumberFormatError
from nsm import word_embeddings
from nsm import data_utils
from utils import load_jsonl, create_envs, collect_traj_for_program, FLAGS


def find_cmd_head(program, pos):
    '''
    find the command name given a position in the whole program
    '''

    if program[pos] == '(':
        return 'abort', 0

    for i in range(pos, -1, -1):
        if program[i] == '(':
            return program[i+1], (pos-i)

    return None, None


def find_entity(namespace, token, type_constraint=None):
    '''
    # first try to determine the type of the entity being numerical or string
    if type_constraint is None:
      try:
        if type(token) == int or type(token) == float:
          val = float(token)
        else:
          val = float(babel.numbers.parse_decimal(token))
        numerical = True
      except NumberFormatError:
        val = normalize(token)
        numerical = False
    else:
      if type_constraint == 'str':
        numerical = False
        val = token if type(token) is str else str(int(token))
      else:
        raise NotImplementedError
    '''
    if type(token) == unicode:
        numerical = False
        val = normalize(token)
    else:
        numerical = True
        val = float(token)

    while True:
        # now we enumerate to find the entity
        for i in range(namespace.n_var-1, -1, -1):
            variable = namespace['v'+str(i)]

            if variable['type'] == 'num_list' and numerical:
                # sanity check
                assert(len(variable['value']) == 1)
                if val == variable['value'][0]:
                    return i, numerical
            elif variable['type'] == 'string_list' and not numerical:
                # sanity check
                assert(len(variable['value']) == 1)
                if val == variable['value'][0]:
                    return i, numerical
            else:
                continue

        #return -1, numerical

        # entity not found -> try the other type
        if not numerical:
            try:
                val = float(babel.numbers.parse_decimal(token))
                numerical = True
            except NumberFormatError:
                return -1, numerical
        else:
            return -1, numerical

    # return -1 when such token is not found as an entity
    return -1, numerical


# ################## copied from preprocess.py ##############################

def normalize(x):
    if not isinstance(x, unicode):
        x = x.decode('utf8', errors='ignore')
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub(ur"[‘’´`]", "'", x)
    x = re.sub(ur"[“”]", "\"", x)
    x = re.sub(ur"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(ur"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(ur"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(ur'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(ur'\s+', ' ', x, flags=re.U).lower().strip()
    return x

# ################## copied from preprocess.py ##############################


def convert(env):
    '''
    Some explanation for the sql language:
        sel: column index (starting from 0)
        agg: aggregation index
        conds: a list of cond

        cond: column_idx, operator_idx, condition

        operator: ['=', '>', '<', 'OP']
        aggregation: ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    '''

    # define some constants to fit the grammar of wikisql
    filters =['filter_eq', 'filter_greater', 'filter_less', 'filter_other', 'filter_eq']
    aggregations = ['none', 'maximum', 'minimum', 'count_stub', 'sum', 'average']

    sql = env.question_annotation['sql']
    code = []

    # first select the rows according to the condition using filters
    for i, cond in enumerate(sql['conds']):
        if i == 0:
            rows = 'all_rows'
        else:
            rows = 'v' + str(env.interpreter.namespace.n_var + i - 1)

        # 1. get the column index
        column = 'v' + str(cond[0])

        # 2. try to locate the token in the entity list and get its type
        entity_var, entity_numerical= find_entity(env.interpreter.namespace, cond[2])
        if entity_var == -1:
            return None
        else:
            value = 'v' + str(entity_var)

        # 3. use the correct filter
        if cond[1] == 0:
            # equal filter, need to know the type
            fltr = filters[0] if entity_numerical else filters[4]
        else:
            fltr = filters[cond[1]]

        statement = ' '.join(['(', fltr, rows, value, column, ')'])
        code.append(statement)

    # then perform the aggregation process
    rows = 'all_rows' if len(sql['conds']) == 0 else ('v' + str(env.interpreter.namespace.n_var+len(sql['conds']) - 1))

    if sql['agg'] == 0: # just hop to the value of the first row
        column = 'v' + str(sql['sel'])
        statement = ' '.join(['(', 'hop', rows, column, ')'])
    else:
        if sql['agg'] == 3: # count, which has slightly different grammar than the other aggs
            statement = ' '.join(['(', 'count', rows, ')'])
        else:
            agg = aggregations[sql['agg']]
            column = 'v' + str(sql['sel'])
            statement = ' '.join(['(', agg, rows, column, ')'])
    code.append(statement)

    # add <END> to the end of the program proper interpretation
    code.append('<END>')
    code = ' '.join(code)

    return code


def get_env_trajs(envs):
    good_oracle_envs = []
    envs_trajs = []
    envs_programs = []

    for i, env in enumerate(envs):
        program = convert(env)
        if program is not None: # means all entities are found

            traj, error_info = collect_traj_for_program(env, program.split(' '), debug=True)

            if traj is None:
                # find the error cmd
                program = error_info[2]
                error_step = len(error_info[1])
                cmd, re_pos = find_cmd_head(program, error_step)
                error_cmd_pos = error_step - re_pos + 1

                # attempt to fix the traj if the interpreter failed to parse
                if cmd == 'filter_eq':
                    # change the entity from the num_list one to string_list one
                    num_entity_idx = int(program[error_cmd_pos+2][1:])
                    str_entity = 'v' + str(num_entity_idx-1)

                    # see if we get the right str entity
                    if env.interpreter.namespace[str_entity]['type'] == 'string_list':
                        program[error_cmd_pos] = 'filter_eq'
                        program[error_cmd_pos+2] = str_entity

                        # see if this simple flip fix this problem
                        traj, error_info = collect_traj_for_program(env, program, debug=True)

            if traj is not None: # means interpreter can successfully parse converted code
                if traj.rewards[-1] == 1.0: # means the code can get reward in the environment
                    good_oracle_envs.append(env)
                    envs_trajs.append(traj)
                    envs_programs.append(program)

    return good_oracle_envs, envs_trajs #, envs_programs


data_folder = "../../data/wikisql/"
train_shard_dir = "../../data/wikisql/processed_input/preprocess_4/"
train_shard_prefix = "train_split_shard_30-"
table_file = "../../data/wikisql/processed_input/preprocess_4/tables.jsonl"
vocab_file = "../../data/wikisql/raw_input/wikisql_glove_vocab.json"
embedding_file = "../../data/wikisql/raw_input/wikisql_glove_embedding_mat.npy"
en_vocab_file = "../../data/wikisql/processed_input/preprocess_4/en_vocab_min_count_5.json"


def get_train_shard_path(i):
    return os.path.join(train_shard_dir, train_shard_prefix  + str(i) + '.jsonl')


def get_envs(env_files=None):
    dataset = []
    if env_files is None:
        fns = [get_train_shard_path(i) for i in range(0, 30)]
    else:
        fns = env_files

    for fn in fns:
        dataset += load_jsonl(fn)
    tables = load_jsonl(table_file)

    table_dict = dict([(table['name'], table) for table in tables])

    # Load pretrained embeddings.
    embedding_model = word_embeddings.EmbeddingModel(vocab_file, embedding_file )

    with open(en_vocab_file, 'r') as f:
        vocab = json.load(f)
    en_vocab = data_utils.Vocab([])
    en_vocab.load_vocab(vocab)

    # Create environments.
    envs = create_envs(table_dict, dataset, en_vocab, embedding_model)

    return envs


def error_analysis():
    f = codecs.open('converter_error_list.bin', 'rb')
    error_list = pickle.load(f)
    f.close()

    # pick the ones that failed to interpret
    interpreter_error_list = [error_example for error_example in error_list if error_example[0] == 'interpreter failed to parse']

    # two set of commands (filter/aggregation) that make the interpreter fail
    filter_list = ['filter_eq', 'filter_greater', 'filter_less', 'filter_str_contain_any']
    agg_list = ['maximum', 'minimum', 'count', 'sum', 'average']
    other_cmd_list = ['hop', 'abort']

    # see for filters/agg, at which argument do they fail to interpret
    filter_error_dict = dict()
    agg_error_dict = dict()
    other_error_dict = dict()
    for filter in filter_list:
        filter_error_dict[filter] = [0,0,0,0,0,0]
    for agg in agg_list:
        agg_error_dict[agg] = [0,0,0,0,0]
    for cmd in other_cmd_list:
        other_error_dict[cmd] = [0,0,0,0,0]

    # see at which step (filter/aggregation) do they fail
    for example in interpreter_error_list:
        program = example[1][2]
        error_step = len(example[1][1])
        cmd, re_pos = find_cmd_head(program, error_step)

        if cmd in filter_list:
            filter_error_dict[cmd][re_pos] += 1
        elif cmd in agg_list:
            agg_error_dict[cmd][re_pos] += 1
        elif cmd in other_cmd_list:
            other_error_dict[cmd][re_pos] += 1
        else:
            print(cmd, program, error_step)
            raise NotImplementedError

    print('Total %d examples can not be interpreted' % len(interpreter_error_list))
    print('%d are filter errors as %s ' % (sum(map(sum, filter_error_dict.values())), filter_error_dict))
    print('%d are aggregation errors as %s ' % (sum(map(sum, agg_error_dict.values())), agg_error_dict))
    print('%d are other errors as %s ' % (sum(map(sum, other_error_dict.values())), other_error_dict))


    # pick the ones that got the wrong answer
    answer_error_list = [error_example for error_example in error_list if error_example[0] == 'wrong answer']

    length_mismatch = 0
    correct_after_normalize = 0
    first_item_match = 0

    for _, env_answer, traj_answer in answer_error_list:
        if len(env_answer) == 0 or len(traj_answer) == 0:
            continue
        if len(env_answer) != len(traj_answer):
            length_mismatch += 1
            if isinstance(env_answer[0], unicode) and isinstance(traj_answer[0], unicode):
                if normalize(env_answer[0]) == normalize(traj_answer[0]):
                    first_item_match+= 1
            else:
                if env_answer[0] == traj_answer[0]:
                    first_item_match+= 1
        else:
            if all([isinstance(answer, unicode) for answer in (env_answer+traj_answer)]):
                normalized_env_answer = [normalize(answer) for answer in env_answer]
                normalized_traj_answer = [normalize(answer) for answer in traj_answer]
                if normalized_env_answer == normalized_traj_answer:
                    correct_after_normalize += 1

    print('Total %d examples got the wrong answer' % len(answer_error_list))
    print('%d are length mismatch but with %d match on the first item' % (length_mismatch, first_item_match))
    print('%d are correct after normalize' % correct_after_normalize)

    return None



def main():
    envs = get_envs()

    # error analysis
    error_list = []
    error_log_file = codecs.open('error_log.txt', 'wb', encoding='utf-8')

    success = 0
    error_1 = 0
    error_2 = 0
    error_3 = 0

    for i, env in tqdm(enumerate(envs)):

        code = convert(env)

        if code is None:
            error_1 += 1
            error_list.append(('can not find entity', None, None))
            continue
        else:
            # verify the correctness of the code
            traj, error_info = collect_traj_for_program(env, code.split(' '), debug=True)

            #'''
            if traj is None:
                # find the error cmd
                program = error_info[2]
                error_step = len(error_info[1])
                cmd, re_pos = find_cmd_head(program, error_step)
                error_cmd_pos = error_step - re_pos + 1

                # attempt to fix the traj if the interpreter failed to parse
                if cmd == 'filter_eq':
                    # change the entity from the num_list one to string_list one
                    num_entity_idx = int(program[error_cmd_pos+2][1:])
                    str_entity = 'v' + str(num_entity_idx-1)

                    # see if we get the right str entity
                    if env.interpreter.namespace[str_entity]['type'] == 'string_list':
                        program[error_cmd_pos] = 'filter_eq'
                        program[error_cmd_pos+2] = str_entity

                        # see if this simple flip fix this problem
                        traj, error_info = collect_traj_for_program(env, program, debug=True)
            #'''

            if traj is not None:
                if traj.rewards[-1] == 1.0:
                    success += 1
                else:
                    error_3 += 1
                    error_list.append(('wrong answer', env.answer, traj.answer))
                    err_log = '%d, expected answer %s, but got answer %s, question is \' %s \' with table %s \n' \
                              % (i, env.answer, traj.answer, env.question_annotation['question'], env.question_annotation['context'])
                    print(err_log)
                    error_log_file.write(err_log)

            else:
                error_2 += 1
                error_list.append(('interpreter failed to parse', error_info, None))

                error_step = len(error_info[1])
                error_token = error_info[2][error_step]
                cmd, re_pos = find_cmd_head(error_info[2], error_step)
                err_log = '%d, command %s step %d error token %s, full program: %s \n' % (i, cmd, error_step, error_token, error_info[2])
                print(err_log)
                error_log_file.write(err_log)

    error_log_file.close()
    print('total %d example, successful converted %d (%f), %d (%f) can not find entity, %d (%f) failed to interpret and %d (%f) got wrong answer'
          % (len(envs),
             success, float(success)/len(envs),
             error_1, float(error_1)/len(envs),
             error_2, float(error_2)/len(envs),
             error_3, float(error_3)/len(envs)))

    with codecs.open('converter_error_list.bin', 'wb') as f:
        pickle.dump(error_list, f)


def loaded_program_analysis():
    envs = get_envs()

    saved_program_file = '../../data/wikisql/processed_input/preprocess_2/all_train_saved_programs-1k_5.json'
    #saved_program_file = '../../data/wikitable/processed_input/all_train_saved_programs.json'
    with open(saved_program_file, 'r') as f:
        program_dict = json.load(f)

    non_empty_env = 0
    spurious_program_enc = 0
    avg_nonempty_env = 0

    for key in program_dict.keys():
        program_list = program_dict[key]
        if len(program_list) > 0:
            non_empty_env += 1
            avg_nonempty_env += len(program_list)
            if len(program_list) > 1:
                spurious_program_enc += 1
    avg_nonempty_env = avg_nonempty_env / float(non_empty_env)

    print '%d items in loaded programs, with %d non-empty and %d have spurious forms with avg of %f' \
          % (len(program_dict), non_empty_env, spurious_program_enc, avg_nonempty_env)

    return

    envs, env_trajs, env_programs = get_env_trajs(envs)

    # stats
    non_empty_env = 0
    match_oracle = 0

    for env, env_traj, program in zip(envs, env_trajs, env_programs):
        env_loaded_program_list = program_dict[env.name]
        if env_loaded_program_list is not None and len(env_loaded_program_list) != 0:

            non_empty_env += 1
            if program in env_loaded_program_list:
                match_oracle += 1

    print '%d items in loaded programs, with %d non-empty and %d envs have an oracle match' % (len(program_dict), non_empty_env, match_oracle)



if __name__ == '__main__':
    FLAGS.executor = 'wikisql'
    #main()
    #error_analysis()

    loaded_program_analysis()

