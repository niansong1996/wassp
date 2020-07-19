from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from math import sqrt

import heapq
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from nsm import data_utils
from nsm import agent_factory
from nsm import word_embeddings

from table.SQL_converter import get_env_trajs
from table.wtq.annotate import get_wtq_annotations
from nsm.agent_factory import traj_to_program
from table.utils import FLAGS, get_sketch, average_token_embedding, init_experiment, get_train_shard_path, json_to_envs
from table.utils import get_init_model_path, get_saved_graph_config, create_agent, collect_traj_for_program


##########################################################################
# the classes of picker are encapsulation of different methods
# to pick the examples for querying annotations
##########################################################################


class ActivePicker:
    @staticmethod
    def parallel_eval(envs, eval_func, process_n=FLAGS.n_actors):
        # prepare the params for creating the agent and splitting the envs
        env_split_size = len(envs) / process_n

        envs_tasks = []
        for i in range(process_n):
            process_envs = envs[i*env_split_size:(i+1)*env_split_size]
            envs_tasks.append(process_envs)

        # distributed evaluation and pick the highest scored examples within budget
        print('Started distributed evaluation with %d processes...' % process_n)
        evaluation_pool = Pool(FLAGS.n_actors)
        all_example_eval_results = evaluation_pool.map(eval_func, envs_tasks)
        evaluation_pool.close()
        evaluation_pool.terminate()
        all_example_eval_results = reduce(lambda x, y: x+y, all_example_eval_results)
        print('Finished distributed evaluation.')

        return all_example_eval_results

    def pick_query_examples(self, envs, budget=FLAGS.al_budget_n):
        '''
        returns a list of picked envs within the budget
        '''
        eval_results = self.eval_examples(envs)
        picked_results = sorted(eval_results, key=lambda x: x[0], reverse=True)
        print('Finished evaluation, max/min score with %s is %.2f/%.2f budget is %d'
              % (str(FLAGS.active_picker_class), picked_results[0][0], picked_results[-1][0], budget))

        return picked_results[:budget]

        scores, env_names = zip(*picked_results)
        prob_sum = sum(scores)
        if budget >= len(env_names):
            picked_env_names = env_names
        else:
            picked_env_names = np.random.choice(env_names, budget, replace=False, p=map(lambda score: score/prob_sum, scores)).tolist()

        env_name_score_dict = dict(map(lambda (score, env_name): (env_name, score), picked_results))
        result = map(lambda env_name: (env_name_score_dict[env_name], env_name), picked_env_names)
        result = sorted(result, key=lambda x: x[0], reverse=True)

        return result

    def eval_examples(self, envs):
        return None


class AllPicker(ActivePicker):
    '''Simply picks all envs'''
    def eval_examples(self, envs):
        rand_perm_envs = list(np.random.permutation(envs))
        rand_perm_envs = [(1.0, env['id']) for env in rand_perm_envs]
        return rand_perm_envs


class WordPicker(ActivePicker):
    '''
    This picker:
    1. evaluates which words are most likely to lead to failure
    2. pick the examples with most of these words
    '''
    def eval_examples(self, envs):
        # first get all the failed examples
        failed_env_names = set([result[1] for result in ActivePicker.parallel_eval(envs, failed_eval)])

        # train a naive bayes on predicting fail/success (1/0)
        train_x, train_y = zip(*map(lambda x: (x['question'], 1 if x['id'] in failed_env_names else 0), envs))
        train_x = TfidfVectorizer().fit_transform(train_x)
        train_y = np.array(train_y)
        classifier = MultinomialNB(alpha=1000.0).fit(train_x, train_y)

        # get the failed examples with the highest failure prob
        x_prob = classifier.predict_proba(train_x)

        env_name_fail_prob = sorted(map(lambda (env, prob): (env['id'], prob[1]), zip(envs, x_prob)), key=lambda x: x[1], reverse=True)
        failed_env_name_fail_prob = filter(lambda x: x[0] in failed_env_names, env_name_fail_prob)
        failed_env_name_fail_prob = map(lambda (x,y): (y,x), failed_env_name_fail_prob)

        '''
        idx_fail_prob = sorted(map(lambda (i, prob): (i, prob[1]), enumerate(x_prob)), key=lambda x: x[1], reverse=True)
        rerank_idx = map(lambda x: x[0], idx_fail_prob)

        rerank_train_x = train_x[rerank_idx]
        rerank_train_y = train_y[rerank_idx]


        total_len = len(rerank_idx)
        total_share = 20
        for i in range(0, total_share):
            start = total_len * i / total_share
            end = total_len * (i+1) / total_share
            print('Training accuracy is %.2f' % classifier.score(rerank_train_x[start:end], rerank_train_y[start:end]))
        '''

        return failed_env_name_fail_prob


def conf_eval(envs):
    return ConfidencePicker.eval(envs)


class ConfidencePicker(ActivePicker):
    '''This picker picks the examples with lowest confidence given by the agent'''
    def eval_examples(self, envs):
        env_eval_results = ActivePicker.parallel_eval(envs, conf_eval)
        env_eval_results = sorted(env_eval_results, key=lambda x: x[0], reverse=True)
        return env_eval_results

    @staticmethod
    def eval(envs):
        # first create the real envs from the jsons
        envs = json_to_envs(envs)

        # create the agent
        graph_config = get_saved_graph_config()
        graph_config['use_gpu'] = False
        graph_config['gpu_id'] = '0'
        init_model_path = get_init_model_path()
        agent = create_agent(graph_config, init_model_path)

        # greedy decode
        beam_samples = []
        env_iterator = data_utils.BatchIterator(dict(envs=envs), shuffle=False, batch_size=FLAGS.eval_batch_size)
        for j, batch_dict in tqdm(enumerate(env_iterator)):
            batch_envs = batch_dict['envs']
            beam_samples += agent.beam_search(batch_envs, beam_size=5)

        # group the samples into beams (because the impl is so bad)
        env_beam_dict = dict()
        for sample in beam_samples:
            env_beam_dict[sample.traj.env_name] = env_beam_dict.get(sample.traj.env_name, []) + [sample]

        # get the highest confidence from the beam of each example
        conf_envs = [(1.0 - max(map(lambda x: x.prob, beam)), env_name) for env_name, beam in env_beam_dict.items()]

        return conf_envs


def failed_conf_eval(envs):
    return FailedConfidencePicker.eval(envs)


class FailedConfidencePicker(ActivePicker):
    '''This picker picks the examples with lowest confidence among the failed examples'''
    def eval_examples(self, envs):
        env_eval_results = ActivePicker.parallel_eval(envs, failed_conf_eval)
        env_eval_results = sorted(env_eval_results, key=lambda x: x[0], reverse=True)
        return env_eval_results

    @staticmethod
    def eval(envs):
        # first create the real envs from the jsons
        envs = json_to_envs(envs)

        # create the agent
        graph_config = get_saved_graph_config()
        graph_config['use_gpu'] = False
        graph_config['gpu_id'] = '0'
        init_model_path = get_init_model_path()
        agent = create_agent(graph_config, init_model_path)

        # greedy decode
        beam_samples = []
        env_iterator = data_utils.BatchIterator(dict(envs=envs), shuffle=False, batch_size=FLAGS.eval_batch_size)
        for j, batch_dict in tqdm(enumerate(env_iterator)):
            batch_envs = batch_dict['envs']
            beam_samples += agent.beam_search(batch_envs, beam_size=5)

        # group the samples into beams (because the impl is so bad)
        env_beam_dict = dict()
        for sample in beam_samples:
            env_beam_dict[sample.traj.env_name] = env_beam_dict.get(sample.traj.env_name, []) + [sample]

        # get the top hyps and find those failed ones
        top_hyps = map(lambda (env_name, beam):
                       (env_name, reduce(lambda s1,s2: s1 if s1.prob > s2.prob else s2, beam)),
                       env_beam_dict.items())
        failed_top_hyps = filter(lambda (env_name, sample): sample.traj.rewards[-1] == 0.0, top_hyps)
        conf_envs = map(lambda (env_name, sample): (sample.prob, env_name), failed_top_hyps)

        return conf_envs


def failed_eval(envs):
    return FailedPicker.eval(envs)


class FailedPicker(ActivePicker):
    ''' Pick the envs which top hyps do not generate the correct answer'''
    def eval_examples(self, envs):
        env_eval_results = ActivePicker.parallel_eval(envs, failed_eval)
        env_eval_results = sorted(env_eval_results, key=lambda x: x[0], reverse=True)
        return env_eval_results

    @staticmethod
    def eval(envs):
        # first create the real envs from the jsons
        envs = json_to_envs(envs)

        # create the agent
        graph_config = get_saved_graph_config()
        graph_config['use_gpu'] = False
        graph_config['gpu_id'] = '0'
        init_model_path = get_init_model_path()
        agent = create_agent(graph_config, init_model_path)

        # greedy decode
        greedy_samples = []
        env_iterator = data_utils.BatchIterator(dict(envs=envs), shuffle=False, batch_size=FLAGS.eval_batch_size)
        for j, batch_dict in tqdm(enumerate(env_iterator)):
            batch_envs = batch_dict['envs']
            greedy_samples += agent.generate_samples(batch_envs, n_samples=1, greedy=True, use_cache=False, filter_error=False)

        env_sample_list = zip(envs, greedy_samples)
        failed_env_sample_list = filter(lambda x: x[1].traj.rewards[-1] < 1.0, env_sample_list)

        failed_envs = [env_sample[0] for env_sample in failed_env_sample_list]
        failed_envs = list(np.random.permutation(failed_envs))
        failed_envs = [(1.0, env.name) for env in failed_envs]

        return failed_envs


class ClusterPicker(ActivePicker):
    '''Pick the examples in the largest cluster of failed examples'''


    def eval_examples(self, envs):
        # declare some constant params
        CLUSTER_NUM = 500
        CLUSTER_SAMPLE_SIZE = 15
        FIRST_N_CLUSTERS = 200
        assert(FIRST_N_CLUSTERS * CLUSTER_SAMPLE_SIZE == 3*FLAGS.al_budget_n)

        # # get failed env names
        # env_eval_results = ActivePicker.parallel_eval(envs, failed_eval)
        # failed_env_names_set = set(map(lambda (score, env_name): env_name,
        #                            filter(lambda (score, env_name): score > 0, env_eval_results)))

        # pure clustering, no failed information
        failed_env_names_set = set(map(lambda env: env['id'], envs))

        # get the questions embedding for every environment
        embedding_model = word_embeddings.EmbeddingModel(FLAGS.vocab_file, FLAGS.embedding_file)
        failed_envs = json_to_envs(filter(lambda env_json: env_json['id'] in failed_env_names_set, envs))

        failed_env_names = map(lambda env: env.name, failed_envs)
        embedding_matrix = preprocessing.normalize(
            np.vstack(map(lambda env: average_token_embedding(env.context[-1], embedding_model), failed_envs)), copy=False)

        # run a k-means++ algorithm on this to get clusters
        print('##################################')
        print('Start running k-means algorithm on %d examples... (this could take a while)' % len(failed_env_names))
        print('##################################')
        labels = KMeans(n_clusters=CLUSTER_NUM, random_state=0).fit(embedding_matrix).labels_
        print('##################################')
        print('K-means running done!')
        print('##################################')

        # put env into clusters and index by name
        env_name_clusters = map(lambda i: np.array(failed_env_names)[labels == i].tolist(), range(CLUSTER_NUM))
        env_name_clusters = sorted(env_name_clusters, key=lambda x: len(x), reverse=True)
        assert(len(env_name_clusters[FIRST_N_CLUSTERS-1]) >= CLUSTER_SAMPLE_SIZE)

        # choose CLUSTER_SAMPLE_SIZE examples from the first FIRST_N_CLUSTERS clusters
        choose_from_clusters = map(lambda cluster: np.random.choice(cluster, CLUSTER_SAMPLE_SIZE, replace=False).tolist(), env_name_clusters[:FIRST_N_CLUSTERS])
        chosen_env_names = set(reduce(lambda x, y: x+y, choose_from_clusters))

        result = map(lambda env: (1.0 if env['id'] in chosen_env_names else 0.0, env['id']), envs)

        '''
        # plot to see the performance
        pca = PCA(n_components=2)
        X_r = pca.fit(embedding_matrix).transform(embedding_matrix)

        plt.figure()

        for i in range(CLUSTER_NUM):
            plt.scatter(X_r[labels == i, 0], X_r[labels == i, 1], color=np.random.rand(3,), label='class_'+str(i))
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('PCA of clusters')
        '''

        return result

##########################################################################
# the classes of different annotations that used in different annotators
##########################################################################


class Annotation:
    def __init__(self, env_name):
        self.env_name = env_name

    def __len__(self):
        raise ValueError('the length function of the base class should never be called, call child instead')

    def get_samples(self, env):
        '''turn this annotation into learnable samples'''
        empty_list = list()
        return empty_list

    def verify_program(self, program):
        '''verify certain (explored) program fits the constraint given by this annotation'''
        pass


class OracleAnnotation(Annotation):
    def __init__(self, env, oracle_program):
        Annotation.__init__(self, env.name)
        self.oracle_program = oracle_program

    def __len__(self):
        return 1

    def get_samples(self, env):
        sample_traj = collect_traj_for_program(env, self.oracle_program)
        return [agent_factory.Sample(sample_traj, prob=1.0)]

    def verify_program(self, program):
        return self.oracle_program == program


class SketchAnnotation(Annotation):
    def __init__(self, env, sketch, samples):
        Annotation.__init__(self, env.name)
        self.sketch = sketch
        self.sketch_programs = [(traj_to_program(sample.traj, env.de_vocab), sample.prob) for sample in samples]

    def __len__(self):
        return len(self.sketch_programs)

    def get_samples(self, env):
        samples = [agent_factory.Sample(traj=collect_traj_for_program(env, program), prob=prob)
                   for program, prob in self.sketch_programs]
        return samples

    def verify_program(self, program):
        return get_sketch(program) == self.sketch

##########################################################################
# the classes of the annotator are encapsulation of different supervision
# for examples that the pickers decided to query
##########################################################################

class ActiveAnnotator:
    '''
    Every annotator gets the environments(questions) that need to be annotated and
    annotate them in its own way and result in different subclass of class Annotation
    '''

    def __init__(self):
        pass

    def annotate_example(self, envs):
        '''
        :param envs: the environemnts that picked to annotate
        :return: a dictionary, with keys being env_names and values being annotations
                 Note: every env must have an annotation or None if such anno is not found
        '''
        pass


class OracleAnnotator(ActiveAnnotator):
    '''
    Give the oracle program as the annotation
    '''
    def __init__(self):
        ActiveAnnotator.__init__(self)
        pass

    def annotate_example(self, envs):
        # first create envs from jsons
        envs = json_to_envs(envs)

        if FLAGS.executor == 'wtq':
            oracle_envs, oracle_trajs = get_wtq_annotations(envs)
        else:
            oracle_envs, oracle_trajs = get_env_trajs(envs)
        oracle_env_programs = [(env.name, traj_to_program(traj, envs[0].de_vocab)) for env, traj in zip(oracle_envs, oracle_trajs)]
        env_name_program_dict = dict(oracle_env_programs)

        env_name_annotation_dict = dict()
        for env in envs:
            program = env_name_program_dict.get(env.name, None)
            if program is not None:
                annotation = OracleAnnotation(env, program)
                env_name_annotation_dict[env.name] = annotation
            else:
                env_name_annotation_dict[env.name] = None

        return env_name_annotation_dict


def sketch_annotate(envs):
    return SketchAnnotator.decode_sketch_program(envs)


class SketchAnnotator(ActiveAnnotator):
    '''
    Provide the sketch for the program
    '''
    def __init__(self):
        ActiveAnnotator.__init__(self)
        pass

    def annotate_example(self, envs):
        #return self.annotate_example_decode(envs, sketch_annotate)
        return self.annotate_example_exploration(envs)

    def annotate_example_decode(self, envs, eval_func, process_n=5):
        # prepare the params for creating the agent and splitting the envs
        env_split_size = len(envs) / process_n

        envs_tasks = []
        for i in range(process_n):
            process_envs = envs[i*env_split_size:(i+1)*env_split_size]
            envs_tasks.append(process_envs)

        # distributed evaluation and pick the highest scored examples within budget
        print('Started distributed sketch annotation with %d processes...' % process_n)
        evaluation_pool = Pool(process_n)
        all_example_eval_results = evaluation_pool.map(eval_func, envs_tasks)
        evaluation_pool.close()
        evaluation_pool.terminate()
        print('Finished distributed annotation.')

        # combine the results
        all_result_dict = dict()
        for result in all_example_eval_results:
            all_result_dict.update(result)

        return all_result_dict

    def annotate_example_exploration(self, envs):
        # first create envs from jsons
        envs = json_to_envs(envs)

        if FLAGS.executor == 'wtq':
            oracle_envs, oracle_trajs = get_wtq_annotations(envs)
        else:
            oracle_envs, oracle_trajs = get_env_trajs(envs)

        oracle_env_programs = [(env.name, traj_to_program(traj, envs[0].de_vocab)) for env, traj in zip(oracle_envs, oracle_trajs)]
        env_name_program_dict = dict(oracle_env_programs)
        env_name_annotation_dict = dict()

        for env in envs:
            program = env_name_program_dict.get(env.name, None)

            if program is None:
                env_name_annotation_dict[env.name] = None
            else:
                sketch = get_sketch(program)
                explored_programs = SketchAnnotator.explore_sketch_programs(env, sketch)
                oracle_trajs = [collect_traj_for_program(env, explored_program) for explored_program in explored_programs]
                samples = [agent_factory.Sample(oracle_traj, prob=1.0) for oracle_traj in oracle_trajs]

                annotation = SketchAnnotation(env, sketch, samples)
                env_name_annotation_dict[env.name] = annotation

        return env_name_annotation_dict

    @staticmethod
    def explore_sketch_programs(env, sketch, max_hyp=1000):
        '''provide a sketch, find all the executable programs fit that sketch'''
        env = env.clone()
        env.use_cache = False

        all_hyps = [(env, env.start_ob)]

        for cmd in sketch:
            # first add the left bracket and the cmd head
            new_all_hyps = []
            for env, ob in all_hyps:
                try:
                    action_index = list(ob[0].valid_indices).index(env.de_vocab.lookup('('))
                    ob, _, _, _ = env.step(action_index)
                    action_index = list(ob[0].valid_indices).index(env.de_vocab.lookup(cmd))
                    ob, _, _, _ = env.step(action_index)

                    new_all_hyps.append((env, ob))
                except ValueError:
                    raise ValueError('This should not happen')
            all_hyps = new_all_hyps

            # then for every possible action, we use a cloned env and step that until the closure of this stmt
            stmt_done = False
            while not stmt_done:
                new_all_hyps = []
                for env, ob in all_hyps:
                    valid_indices = list(ob[0].valid_indices)
                    if len(valid_indices) == 0: # this is a dead end for current env
                        continue
                    for action_index in range(len(valid_indices)):
                        new_env = env.clone()
                        new_env.use_cache = False

                        ob, _, _, _ = new_env.step(action_index)
                        new_all_hyps.append((new_env, ob))
                    if valid_indices == [env.de_vocab.lookup(')')]: # maximum stmt length is reached
                        stmt_done = True
                all_hyps = list(np.random.permutation(new_all_hyps))[:max_hyp]

        # add then end token
        new_all_hyps = []
        for env, ob in all_hyps:
            try:
                action_index = list(ob[0].valid_indices).index(env.de_vocab.lookup('<END>'))
                ob, _, _, _ = env.step(action_index)

                new_all_hyps.append((env, ob))
            except ValueError:
                raise ValueError('This should not happen')
        all_hyps = new_all_hyps

        # pruning based on the result
        explored_programs = []
        for env, _ in all_hyps:
            if env.rewards[-1] == 1.0:
                traj = agent_factory.Traj(obs=env.obs, actions=env.actions, rewards=env.rewards,
                                          context=env.get_context(), env_name=env.name, answer=env.interpreter.result)
                explored_programs.append(traj_to_program(traj, env.de_vocab))

        explored_programs = list(np.random.permutation(explored_programs))

        return explored_programs

    @staticmethod
    def decode_sketch_program(envs):
        # first create the real envs from the jsons and add constraints to them
        envs = json_to_envs(envs)
        env_name_dict = dict(map(lambda env: (env.name, env), envs))

        if FLAGS.executor == 'wtq':
            oracle_envs, oracle_trajs = get_wtq_annotations(envs)
        else:
            oracle_envs, oracle_trajs = get_env_trajs(envs)

        env_sketch_dict = dict([(env.name, get_sketch(traj_to_program(traj, envs[0].de_vocab))) for env, traj in zip(oracle_envs, oracle_trajs)])
        for env in envs:
            sketch = env_sketch_dict.get(env.name, None)
            if sketch is not None:
                env.set_sketch_constraint(sketch[:])

        # create the agent
        graph_config = get_saved_graph_config()
        graph_config['use_gpu'] = False
        graph_config['gpu_id'] = '0'
        init_model_path = get_init_model_path()
        agent = create_agent(graph_config, init_model_path)

        # beam search
        beam_samples = []
        env_iterator = data_utils.BatchIterator(dict(envs=envs), shuffle=False, batch_size=FLAGS.eval_batch_size)
        for j, batch_dict in tqdm(enumerate(env_iterator)):
            batch_envs = batch_dict['envs']
            beam_samples += agent.beam_search(batch_envs, beam_size=50)

        # group the samples into beams (because the impl is so bad)
        env_beam_dict = dict()
        for sample in beam_samples:
            env_beam_dict[sample.traj.env_name] = env_beam_dict.get(sample.traj.env_name, []) + [sample]

        # get the trajs with 1.0 reward for each example and re-weight the prob
        env_name_annotation_dict = dict()
        for env_name, env in env_name_dict.iteritems():
            beam = env_beam_dict.get(env_name, [])
            success_beam = filter(lambda x: x.traj.rewards[-1] == 1.0, beam)
            if len(success_beam) > 0:
                # retrieve the sketch result from previous steps
                sketch = env_sketch_dict.get(env_name, None)

                if sketch is None:
                    env_name_annotation_dict[env_name] = None
                else:
                    # re-weight the examples in the beam
                    prob_sum = sum(map(lambda sample: sample.prob, success_beam))
                    success_beam = map(lambda sample: agent_factory.Sample(traj=sample.traj, prob=sample.prob/prob_sum), success_beam)
                    if len(success_beam) > 10:
                        success_beam = sorted(success_beam, key=lambda sample: sample.prob, reverse=True)
                        success_beam = success_beam[:10]

                    annotation = SketchAnnotation(env, sketch, success_beam)
                    env_name_annotation_dict[env_name] = annotation
            else:
                env_name_annotation_dict[env_name] = None

        return env_name_annotation_dict

##########################################################################
# Some more helper functions
##########################################################################
def get_active_picker(picker_name):
    return eval(picker_name)()


def active_learning(envs, picker_name, annotator_name, budget=FLAGS.al_budget_n):
    # print basic info about this active learning
    print('Received %d envs for active learning with %s and %s in a total budget of %d'
          % (len(envs), str(picker_name), str(annotator_name), budget))

    # eval the picker and annotator name and create their class
    picker = eval(picker_name)()
    annotator = eval(annotator_name)()

    # pick more envs than budget in case some of those picked can not be annotated
    if FLAGS.executor == 'wtq':
      safe_budget = 1000000
    else:
      safe_budget = budget*3
      
    picked_results = picker.pick_query_examples(envs, budget=safe_budget)
    picked_env_names = set([result[1] for result in picked_results])
    picked_envs = filter(lambda x: x['id'] in picked_env_names, envs)
    print('Real budget is %d, but picked %d examples to annotate with highest/lowest score as %.2f/%.2f'
          % (budget, safe_budget, picked_results[0][0], picked_results[-1][0]))

    # returns a dict of env_name and its annotation
    annotated_env_dict = annotator.annotate_example(picked_envs)
    good_annotation_n = len(filter(lambda x: x is not None, annotated_env_dict.values()))
    print('Total %d envs sent for annotation, %d got good annotations'
          % (len(picked_envs), good_annotation_n))

    # final cutoff with the real budget
    picked_annotations = dict()
    for env in picked_envs:
        annotation = annotated_env_dict[env['id']]
        if annotation is not None and len(annotation) > 0:
            picked_annotations[env['id']] = annotation
        if len(picked_annotations) >= budget:
            break

    print('Final number of picked annotations: %d (budget %d)' % (len(picked_annotations), budget))
    time.sleep(30)
    return picked_annotations

