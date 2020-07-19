'''
create human annotations for wtq dataset

1. show the table, question and the expected answer (and preferably the predictions for correction)
2. prompt: ask for an annotation (gold MR)
3. transform the annotation to executable form
4. execute it and see if it execute to the gold denotations
'''
import os
import pickle
import json
import numpy as np

from tqdm import tqdm
from nsm import data_utils, word_embeddings
from table.table_utils import display_table
from table.utils import FLAGS, load_envs_as_json, get_train_shard_path, json_to_envs, load_jsonl, create_envs
from table.utils import get_saved_graph_config, get_init_model_path, create_agent, collect_traj_for_program

# file locations

DATA_DIR = os.path.expanduser('~/projects/data/wikitable/')
saved_annotations_file = './wtq_annotations.bin'

# set the FLAGS
'''
FLAGS.n_actors = 1
FLAGS.shard_start = 0
FLAGS.shard_end = 90
FLAGS.max_n_mem = 60
FLAGS.max_n_valid_indices = 60
FLAGS.table_file = os.path.join(DATA_DIR, 'processed_input/preprocess_14/tables.jsonl')
FLAGS.train_shard_dir = os.path.join(DATA_DIR, 'processed_input/preprocess_14/data_split_1/')
FLAGS.train_shard_prefix = 'train_split_shard_90-'
FLAGS.embedding_file = os.path.join(DATA_DIR, 'raw_input/wikitable_glove_embedding_mat.npy')
FLAGS.vocab_file = os.path.join(DATA_DIR, 'raw_input/wikitable_glove_vocab.json')
FLAGS.load_model_from_experiment = os.path.join(DATA_DIR, 'output/baseline')
FLAGS.output_dir = os.path.join(DATA_DIR, 'output')
FLAGS.init_model_path = os.path.join(DATA_DIR, 'output/baseline/best_model/model-14619')
FLAGS.active_picker_class = 'FailedPicker'
FLAGS.en_vocab_file = os.path.join(DATA_DIR, 'processed_input/preprocess_14/en_vocab_min_count_5.json')
'''


def verify_annotation(env, program):
    # simply check the reward for the program
    traj, error_info = collect_traj_for_program(env, program, debug=True)

    if traj is None:
        valid_tokens = map(lambda x: env.de_vocab.lookup(x, reverse=True), error_info[4])
        input_token = error_info[2][len(error_info[1])]
        print '###Program not executable on this environment, valid tokens are (%s) but get (%s)###' % (valid_tokens, input_token)
        return False
    elif traj.rewards[-1] != 1.0:
        print '###Program not executed to the expected answer, expected(%s) but get(%s)###' % (env.answer, traj.answer)
        return False
    else:
        return True


def annotate(env, table):
    # first print the table
    try:
        display_table(table)
    except:
        print 'Table Format Error, jump to next example'
        return ['table format error']

    # show intermediate variables
    for var_i in range(env.interpreter.namespace.n_var):
        var_name = 'v'+str(var_i)
        print var_name, env.interpreter.namespace[var_name]

    # then show the question and expected result
    print('Question: %s' % env.question_annotation['question'])
    print('Expected Answer: %s' % env.question_annotation['answer'])

    while True:
        # ask for the annotation and convert to executable form
        user_input = raw_input('Input the annotation:').strip()
        program = user_input.split(' ')

        if program == ['skip']:
            print 'Annotator choose to skip this example!'
            return program
        else:
            if program[-1] != '<END>':
                program.append('<END>')

            if verify_annotation(env, program):
                print 'Correct Annotation!'
                return program
            else:
                print 'Incorrect Annotation! (See Above)'


def get_examples_to_annotate():
    ''' Operates in two cases:
        1. if a saved list to load, load the list
        2. if no such list, load the model, the training set and re-evaluate
    :return: an ORDERED (by importance) list of environments (training examples) to annotate
    '''

    if os.path.exists(saved_annotations_file):
        # simply load and continue annotating
        with open(saved_annotations_file, 'r') as f:
            annotation_result_list = pickle.load(f)
    else:
        # get the training examples (environments)
        envs = load_envs_as_json([get_train_shard_path(i) for i in range(FLAGS.shard_start, FLAGS.shard_end)])
        env_name_dict = dict(map(lambda env: (env['id'], env), envs))

        # pick the examples for annotations
        from table.active_learning import get_active_picker
        picker = get_active_picker(FLAGS.active_picker_class)
        picked_results = picker.pick_query_examples(envs, budget=len(envs))

        # create a list of (env_json, score, annotation) and save it
        annotation_result_list = map(lambda (score, env_name):
                                     (env_name_dict[env_name], score, None),
                                     picked_results)

        with open(saved_annotations_file, 'w+') as f:
            pickle.dump(annotation_result_list, f)

    return annotation_result_list


def sync_results(annotation_result_list):
    with open(saved_annotations_file, 'wb') as f:
        pickle.dump(annotation_result_list, f)


def get_wtq_annotations(envs):
    annotation_result_list = get_examples_to_annotate()
    env_name_program_dict = dict(map(lambda (env_json, _, program):
                                (env_json['id'], program if program is None or program[0] == '(' else None), annotation_result_list))
    good_envs = []
    good_env_trajs = []
    for env in envs:
        env_name = env.name
        env_program = env_name_program_dict.get(env_name, None)
        if env_program is not None and verify_annotation(env, env_program):
            good_envs.append(env)
            good_env_trajs.append(collect_traj_for_program(env, env_program))

    return good_envs, good_env_trajs


def separate_annotation_file(n_files=2):
    annotation_result_list = get_examples_to_annotate()

    result_lists = map(lambda x: list(), range(n_files))

    valid_annotation_counter = 0
    for result in annotation_result_list:
        result_lists = map(lambda result_list: result_list + [(result[0], result[1], None)], result_lists)
        if result[2] is not None and result[2][0] == '(' and valid_annotation_counter < 50:
            result_lists[valid_annotation_counter % n_files][-1] = (result[0], result[1], result[2])
            valid_annotation_counter += 1

    for i, result_list in enumerate(result_lists):
        file_name = saved_annotations_file.split('.')
        file_name[-2] += '_%d' % i
        file_name = '.'.join(file_name)

        with open(file_name, 'wb') as f:
            pickle.dump(result_lists[i], f)






def main():
    '''
    1. call get_examples_to_annotate() to get an ordered list of examples to annotated
    4. call annotate() on these examples and get the VERIFIED annotation
    5. call sync_result() to save the results
    :return: None, all results are locally saved in files
    '''

    # load the tables
    tables = load_jsonl(FLAGS.table_file)
    table_dict = dict([(table['name'], table) for table in tables])

    # Load pre-trained embeddings.
    embedding_model = word_embeddings.EmbeddingModel(
        FLAGS.vocab_file, FLAGS.embedding_file)

    with open(FLAGS.en_vocab_file, 'r') as f:
        vocab = json.load(f)
    en_vocab = data_utils.Vocab([])
    en_vocab.load_vocab(vocab)

    annotation_result_list = get_examples_to_annotate()
    for i in range(len(annotation_result_list)):
        if annotation_result_list[i][2] is None:
            # create a real environment
            env = create_envs(table_dict, [annotation_result_list[i][0]], en_vocab, embedding_model)[0]

            # get the annotation
            annotation = annotate(env, table_dict[env.question_annotation['context']])
            annotation_result_list[i] = (annotation_result_list[i][0], annotation_result_list[i][1], annotation)

            sync_results(annotation_result_list)



if __name__ == '__main__':
    #separate_annotation_file()
    main()
