import json
import copy

from tqdm import tqdm

from collections import namedtuple

from table.utils import collect_traj_for_program, get_train_shard_path
from table.utils import FLAGS, wikisql_score, get_statements, get_sketch
from nsm.agent_factory import traj_to_program
from SQL_converter import get_envs, get_env_trajs


detailed_example_result = namedtuple('detailed_example_result', ['id', 'table', 'question', 'oracle_program', 'oracle_answer', 'hyps_in_beam'])
program_result = namedtuple('program_result', ['prob', 'predicted_program', 'predicted_answer'])


def is_same_answer(predicted, oracle):
    return bool(wikisql_score(predicted, oracle))


def is_same_length(predicted, oracle):
    return len(get_sketch(predicted)) == len(get_sketch(oracle))


def is_same_sketch(predicted, oracle):
    return get_sketch(predicted) == get_sketch(oracle)


def is_same_program(predicted, oracle):
    return predicted == oracle


def confidence_analysis(example_results):
    top_90_plus = 0
    top_80_90 = 0
    top_70_80 = 0
    top_70_minus = 0

    for example in example_results:
        beam_confidence = [hyp.prob for hyp in example.hyps_in_beam]
        top_confidence = beam_confidence[0]
        if top_confidence > 0.9:
            top_90_plus += 1
        elif top_confidence > 0.8:
            top_80_90 += 1
        elif top_confidence > 0.7:
            top_70_80 += 1
        else:
            top_70_minus += 1

    normalize_func = lambda x: (x * 100.0) / float(len(example_results))
    print(('%d examples in total, measuring top hyp confidence: '
          '%.2f confidence 0.9+, %.2f confidence 0.8-0.9, '
          '%.2f confidence 0.7-0.8, %.2f confidence 0.7-') %
          (len(example_results),
           normalize_func(top_90_plus), normalize_func(top_80_90),
           normalize_func(top_70_80), normalize_func(top_70_minus)))


def overall_analysis(example_results):
    correct_length_ratio = [0] * len(example_results[0].hyps_in_beam)
    correct_sketch_ratio = [0] * len(example_results[0].hyps_in_beam)
    correct_answer_ratio = [0] * len(example_results[0].hyps_in_beam)
    correct_program_ratio = [0] * len(example_results[0].hyps_in_beam)

    for example_result in example_results:
        oracle_program = example_result.oracle_program
        oracle_answer = example_result.oracle_answer

        for i, hyp in enumerate(example_result.hyps_in_beam):
            correct_length_ratio[i] += is_same_length(hyp.predicted_program, oracle_program)
            correct_sketch_ratio[i] += is_same_sketch(hyp.predicted_program, oracle_program)
            correct_answer_ratio[i] += is_same_answer(hyp.predicted_answer, oracle_answer)
            correct_program_ratio[i] += is_same_program(hyp.predicted_answer, oracle_answer)

    # output the analysis result
    normalize_func = lambda x: x / float(len(example_results))
    print('correct stmt length ratio for hyps in beam are %s' % str(map(normalize_func, correct_length_ratio)))
    print('correct program sketch ratio for hyps in beam are %s' % str(map(normalize_func, correct_sketch_ratio)))
    print('correct execution answer ratio for hyps in beam are %s' % str(map(normalize_func, correct_answer_ratio)))
    print('exactly the same program ratio for hyps in beam are %s' % str(map(normalize_func, correct_program_ratio)))


def failed_words_analysis(all_results, failed_results):
    '''
    Count the most frequent words
    '''
    all_word_dict = dict()
    failed_word_dict = dict()
    successful_word_dict = dict()

    for example in all_results:
        for word in example.question.split(' '):
            all_word_dict[word] = all_word_dict.get(word, 0) + 1
            successful_word_dict[word] = successful_word_dict.get(word, 0) + 1

    for example in failed_results:
        for word in example.question.split(' '):
            failed_word_dict[word] = failed_word_dict.get(word, 0) + 1
            successful_word_dict[word] = successful_word_dict[word] - 1

    failed_result_list = []
    successful_result_list = []

    # ditch the failed and successful words with low frequency
    failed_word_dict = dict(filter(lambda x: x[1] > 20, failed_word_dict.iteritems()))
    successful_word_dict = dict(filter(lambda x: x[1] > 20, successful_word_dict.iteritems()))

    for word in failed_word_dict.keys():
        failed_count = failed_word_dict[word]
        all_count = all_word_dict[word]
        ratio = float(failed_count) / float(all_count)
        failed_result_list.append((word, ratio, failed_count, all_count))

    for word in successful_word_dict.keys():
        successful_count = successful_word_dict[word]
        all_count = all_word_dict[word]
        ratio = float(successful_count) / float(all_count)
        successful_result_list.append((word, ratio, successful_count, all_count))

    failed_result_list = sorted(failed_result_list, key=lambda x: x[1], reverse=True)
    successful_result_list = sorted(successful_result_list, key=lambda x: x[1], reverse=True)

    print('top words lead to failure:')
    for result in failed_result_list:
        if result[0].isalpha():
            print('%s : %.2f (%d/%d)' % (result[0], result[1], result[2], result[3]))

    print('top words lead to success:')
    for result in successful_result_list:
        if result[0].isalpha():
            print('%s : %.2f (%d/%d)' % (result[0], result[1], result[2], result[3]))


def program_detailed_error_analysis(example_results):
    extra_stmt = 0
    missing_filter = 0
    missing_aggregation = 0

    filter_mismatch = 0
    aggregation_mismatch = 0
    filter_aggr_mismatch = 0

    filter_cmd = 0
    filter_target = 0
    filter_entity = 0
    filter_column = 0

    aggr_cmd = 0
    aggr_target = 0
    aggr_column = 0

    for example in example_results:
        predicted_stmts = get_statements(example.hyps_in_beam[0].predicted_program)
        oracle_stmts = get_statements(example.oracle_program)

        # comparing the length
        if len(predicted_stmts) != len(oracle_stmts):
            if len(predicted_stmts) > len(oracle_stmts):
                extra_stmt += 1
            else:
                if len(predicted_stmts[-1]) > 5:
                    missing_aggregation += 1
                else:
                    missing_filter += 1
            continue

        # count how many statements mismatch
        stmt_mismatch = []
        for i, oracle_stmt in enumerate(oracle_stmts):
            # try to find the corresponding stmt in the predicted program
            found = False
            for predicted_stmt in predicted_stmts:
                if oracle_stmt == predicted_stmt:
                    found = True
                    break
            if not found:
                stmt_mismatch.append(i)









    pass


def wikisql_error_analysis():
    env_file = '/Users/ansongni/projects/data/wikisql/processed_input/preprocess_4/test_split.jsonl'
    #train_env_file_prefix = '/Users/ansongni/projects/data/wikisql/processed_input/preprocess_4/train_split_shard_30-'
    decoded_beam_file = '/Users/ansongni/projects/data/wikisql/output/eval_imp_baseline/dev_programs_in_beam_0.json'
    #decoded_beam_file = '/Users/ansongni/projects/data/wikisql/output/train_eval_imp_baseline/dev_programs_in_beam_0.json'

    # first load the test environments and get oracle programs
    #test_envs = get_envs([(train_env_file_prefix+str(i)+'.jsonl') for i in range(0, 30)])
    test_envs = get_envs([env_file])
    envs, trajs = get_env_trajs(test_envs)

    oracle_env_programs = [(env, traj_to_program(traj, envs[0].de_vocab)) for env, traj in zip(envs, trajs)]

    # then load decoded results in the beam
    with open(decoded_beam_file) as f:
        decoded_beam = json.load(f)

    example_results = []

    # generate the detailed example result for each env that got oracle
    for env, oracle_program in oracle_env_programs:
        id = env.name
        table = env.question_annotation['context']
        question = env.question_annotation['question']
        oracle_answer = env.question_annotation['answer']

        # take care of missing example
        hyps = decoded_beam.get(id, None)
        if hyps is None:
            continue

        beam = []
        for hyp in hyps:
            prob = hyp[2]
            predicted_program = hyp[0]
            predicted_answer = hyp[1]

            program_hyp = program_result(prob, predicted_program, predicted_answer)
            beam.append(program_hyp)

        result = detailed_example_result(id, table, question, oracle_program, oracle_answer, beam)
        example_results.append(result)

    # 1. now we do overall error analysis
    print('%d test examples, %d have oracle program, %d gets evaluated' % (len(test_envs), len(envs), len(example_results)))
    overall_analysis(example_results)
    confidence_analysis(example_results)

    # 2. now we analyze the failed cases
    failed_example_results = filter(lambda result: 1.0 - wikisql_score(result.hyps_in_beam[0].predicted_answer, result.oracle_answer), example_results)
    print('%d failed examples' % len(failed_example_results))
    overall_analysis(failed_example_results)
    confidence_analysis(failed_example_results)
    failed_words_analysis(example_results, failed_example_results)

    #for result in failed_example_results:
    #    print('predicted: %s \n  oracle: %s \n' % (result.hyps_in_beam[0].predicted_program, result.oracle_program))
    # TODO: 1. only analysis for failed cases; 2. analyze the percentage of not confident examples


def beam_against_oracle(beam, oracle):
    '''
    :param beam: hypothesis program(s) in the beam
    :param oracle: oracle program given by the sql_coverter
    :return: stats
    '''

    ''' need to measure:
        1. How many of them get 1.0 reward
        2. How many of them get 1.0 reward but is spurious
        3. Any one got oracle program
        4. If not, why it that the case? (wrong command? wrong column? wrong entity?)
    '''

    pass




if __name__ == '__main__':
    FLAGS.executor = 'wikisql'
    wikisql_error_analysis()
