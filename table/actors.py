import multiprocessing
import os
import codecs
import copy
import time

import tensorflow as tf

from nsm import agent_factory
from nsm import data_utils

from table.utils import init_experiment, FLAGS, get_train_shard_path, get_init_model_path, load_programs
from table.utils import get_experiment_dir, show_samples, collect_traj_for_program
from table.SQL_converter import get_env_trajs


class Actor(multiprocessing.Process):

    def __init__(
            self, name, actor_id, shard_ids, ckpt_queue, train_queue, eval_queue, replay_queue):
        multiprocessing.Process.__init__(self)
        self.ckpt_queue = ckpt_queue
        self.eval_queue = eval_queue
        self.train_queue = train_queue
        self.replay_queue = replay_queue
        self.name = name
        self.shard_ids = shard_ids
        self.actor_id = actor_id

    def run(self):
        agent, envs = init_experiment(
            [get_train_shard_path(i) for i in self.shard_ids],
            use_gpu=FLAGS.actor_use_gpu,
            gpu_id=str(self.actor_id + FLAGS.actor_gpu_start_id))

        # only keep the envs that can get oracle traj for a fair compare
        #envs, _ = get_env_trajs(envs)

        graph = agent.model.graph
        current_ckpt = get_init_model_path()

        env_dict = dict([(env.name, env) for env in envs])
        replay_buffer = agent_factory.AllGoodReplayBuffer(agent, envs[0].de_vocab)

        # Load saved programs to warm start the replay buffer.
        if FLAGS.load_saved_programs:
            load_programs(
                envs, replay_buffer, FLAGS.saved_program_file)

        if FLAGS.save_replay_buffer_at_end:
            replay_buffer_copy = agent_factory.AllGoodReplayBuffer(de_vocab=envs[0].de_vocab)
            replay_buffer_copy.program_prob_dict = copy.deepcopy(replay_buffer.program_prob_dict)

        i = 0
        while True:
            # Create the logging files.
            if FLAGS.log_samples_every_n_epoch > 0 and i % FLAGS.log_samples_every_n_epoch == 0:
                f_replay = codecs.open(os.path.join(
                    get_experiment_dir(), 'replay_samples_{}_{}.txt'.format(self.name, i)),
                    'w', encoding='utf-8')
                f_policy = codecs.open(os.path.join(
                    get_experiment_dir(), 'policy_samples_{}_{}.txt'.format(self.name, i)),
                    'w', encoding='utf-8')
                f_train = codecs.open(os.path.join(
                    get_experiment_dir(), 'train_samples_{}_{}.txt'.format(self.name, i)),
                    'w', encoding='utf-8')

            n_train_samples = 0
            if FLAGS.use_replay_samples_in_train:
                n_train_samples += FLAGS.n_replay_samples

            if FLAGS.use_policy_samples_in_train and FLAGS.use_nonreplay_samples_in_train:
                raise ValueError(
                    'Cannot use both on-policy samples and nonreplay samples for training!')

            if FLAGS.use_policy_samples_in_train or FLAGS.use_nonreplay_samples_in_train:
                # Note that nonreplay samples are drawn by rejection
                # sampling from on-policy samples.
                n_train_samples += FLAGS.n_policy_samples

            # Make sure that all the samples from the env batch
            # fits into one batch for training.
            if FLAGS.batch_size < n_train_samples:
                raise ValueError(
                    'One batch have to at least contain samples from one environment.')

            env_batch_size = FLAGS.batch_size / n_train_samples

            env_iterator = data_utils.BatchIterator(
                dict(envs=envs), shuffle=True,
                batch_size=env_batch_size)

            for j, batch_dict in enumerate(env_iterator):
                batch_envs = batch_dict['envs']
                tf.logging.info('=' * 50)
                tf.logging.info('{} iteration {}, batch {}: {} envs'.format(
                    self.name, i, j, len(batch_envs)))

                t1 = time.time()
                # Generate samples with cache and save to replay buffer.
                t3 = time.time()
                n_explore = 0
                for _ in xrange(FLAGS.n_explore_samples):
                    explore_samples = agent.generate_samples(
                        batch_envs, n_samples=1, use_cache=FLAGS.use_cache,
                        greedy=FLAGS.greedy_exploration)
                    replay_buffer.save(explore_samples)
                    n_explore += len(explore_samples)

                if FLAGS.n_extra_explore_for_hard > 0:
                    hard_envs = [env for env in batch_envs
                                 if not replay_buffer.has_found_solution(env.name)]
                    if hard_envs:
                        for _ in xrange(FLAGS.n_extra_explore_for_hard):
                            explore_samples = agent.generate_samples(
                                hard_envs, n_samples=1, use_cache=FLAGS.use_cache,
                                greedy=FLAGS.greedy_exploration)
                            replay_buffer.save(explore_samples)
                            n_explore += len(explore_samples)

                t4 = time.time()
                tf.logging.info('{} sec used generating {} exploration samples.'.format(
                    t4 - t3, n_explore))

                tf.logging.info('{} samples saved in the replay buffer.'.format(
                    replay_buffer.size))

                t3 = time.time()
                replay_samples = replay_buffer.replay(
                    batch_envs, FLAGS.n_replay_samples,
                    use_top_k=FLAGS.use_top_k_replay_samples,
                    agent=None if FLAGS.random_replay_samples else agent,
                    truncate_at_n=FLAGS.truncate_replay_buffer_at_n)
                t4 = time.time()
                tf.logging.info('{} sec used selecting {} replay samples.'.format(
                    t4 - t3, len(replay_samples)))

                t3 = time.time()
                if FLAGS.use_top_k_policy_samples:
                    if FLAGS.n_policy_samples == 1:
                        policy_samples = agent.generate_samples(
                            batch_envs, n_samples=FLAGS.n_policy_samples,
                            greedy=True)
                    else:
                        policy_samples = agent.beam_search(
                            batch_envs, beam_size=FLAGS.n_policy_samples)
                else:
                    policy_samples = agent.generate_samples(
                        batch_envs, n_samples=FLAGS.n_policy_samples,
                        greedy=False)
                t4 = time.time()
                tf.logging.info('{} sec used generating {} on-policy samples'.format(
                    t4-t3, len(policy_samples)))

                t2 = time.time()
                tf.logging.info(
                    ('{} sec used generating replay and on-policy samples,'
                     ' {} iteration {}, batch {}: {} envs').format(
                        t2-t1, self.name, i, j, len(batch_envs)))

                t1 = time.time()
                self.eval_queue.put((policy_samples, len(batch_envs)))
                self.replay_queue.put((replay_samples, len(batch_envs)))

                assert (FLAGS.fixed_replay_weight >= 0.0 and FLAGS.fixed_replay_weight <= 1.0)

                if FLAGS.use_replay_prob_as_weight:
                    new_samples = []
                    for sample in replay_samples:
                        name = sample.traj.env_name
                        if name in replay_buffer.prob_sum_dict:
                            replay_prob = max(
                                replay_buffer.prob_sum_dict[name], FLAGS.min_replay_weight)
                        else:
                            replay_prob = 0.0
                        scale = replay_prob
                        new_samples.append(
                            agent_factory.Sample(
                                traj=sample.traj,
                                prob=sample.prob * scale))
                    replay_samples = new_samples
                else:
                    replay_samples = agent_factory.scale_probs(
                        replay_samples, FLAGS.fixed_replay_weight)

                replay_samples = sorted(
                    replay_samples, key=lambda x: x.traj.env_name)

                policy_samples = sorted(
                    policy_samples, key=lambda x: x.traj.env_name)

                if FLAGS.use_nonreplay_samples_in_train:
                    nonreplay_samples = []
                    for sample in policy_samples:
                        if not replay_buffer.contain(sample.traj):
                            nonreplay_samples.append(sample)

                replay_buffer.save(policy_samples)

                def weight_samples(samples):
                    if FLAGS.use_replay_prob_as_weight:
                        new_samples = []
                        for sample in samples:
                            name = sample.traj.env_name
                            if name in replay_buffer.prob_sum_dict:
                                replay_prob = max(
                                    replay_buffer.prob_sum_dict[name],
                                    FLAGS.min_replay_weight)
                            else:
                                replay_prob = 0.0
                            scale = 1.0 - replay_prob
                            new_samples.append(
                                agent_factory.Sample(
                                    traj=sample.traj,
                                    prob=sample.prob * scale))
                    else:
                        new_samples = agent_factory.scale_probs(
                            samples, 1 - FLAGS.fixed_replay_weight)
                    return new_samples

                train_samples = []
                if FLAGS.use_replay_samples_in_train:
                    if FLAGS.use_trainer_prob:
                        replay_samples = [
                            sample._replace(prob=None) for sample in replay_samples]
                    train_samples += replay_samples

                if FLAGS.use_policy_samples_in_train:
                    train_samples += weight_samples(policy_samples)

                if FLAGS.use_nonreplay_samples_in_train:
                    train_samples += weight_samples(nonreplay_samples)

                train_samples = sorted(train_samples, key=lambda x: x.traj.env_name)
                tf.logging.info('{} train samples'.format(len(train_samples)))

                if FLAGS.use_importance_sampling:
                    step_logprobs = agent.compute_step_logprobs(
                        [s.traj for s in train_samples])
                else:
                    step_logprobs = None

                if FLAGS.use_replay_prob_as_weight:
                    n_clip = 0
                    for env in batch_envs:
                        name = env.name
                        if (name in replay_buffer.prob_sum_dict and
                                replay_buffer.prob_sum_dict[name] < FLAGS.min_replay_weight):
                            n_clip += 1
                    clip_frac = float(n_clip) / len(batch_envs)
                else:
                    clip_frac = 0.0

                self.train_queue.put((train_samples, step_logprobs, clip_frac))
                t2 = time.time()
                tf.logging.info(
                    ('{} sec used preparing and enqueuing samples, {}'
                     ' iteration {}, batch {}: {} envs').format(
                        t2-t1, self.name, i, j, len(batch_envs)))

                t1 = time.time()
                # Wait for a ckpt that still exist or it is the same
                # ckpt (no need to load anything).
                while True:
                    new_ckpt = self.ckpt_queue.get()
                    new_ckpt_file = new_ckpt + '.meta'
                    if new_ckpt == current_ckpt or tf.gfile.Exists(new_ckpt_file):
                        break
                t2 = time.time()
                tf.logging.info('{} sec waiting {} iteration {}, batch {}'.format(
                    t2-t1, self.name, i, j))

                if new_ckpt != current_ckpt:
                    # If the ckpt is not the same, then restore the new
                    # ckpt.
                    tf.logging.info('{} loading ckpt {}'.format(self.name, new_ckpt))
                    t1 = time.time()
                    graph.restore(new_ckpt)
                    t2 = time.time()
                    tf.logging.info('{} sec used {} restoring ckpt {}'.format(
                        t2-t1, self.name, new_ckpt))
                    current_ckpt = new_ckpt

                if FLAGS.log_samples_every_n_epoch > 0 and i % FLAGS.log_samples_every_n_epoch == 0:
                    f_replay.write(show_samples(replay_samples, envs[0].de_vocab, env_dict))
                    f_policy.write(show_samples(policy_samples, envs[0].de_vocab, env_dict))
                    f_train.write(show_samples(train_samples, envs[0].de_vocab, env_dict))

            if FLAGS.log_samples_every_n_epoch > 0 and i % FLAGS.log_samples_every_n_epoch == 0:
                f_replay.close()
                f_policy.close()
                f_train.close()

            if agent.model.get_global_step() >= FLAGS.n_steps:
                if FLAGS.save_replay_buffer_at_end:
                    all_replay = os.path.join(get_experiment_dir(),
                                              'all_replay_samples_{}.txt'.format(self.name))
                with codecs.open(all_replay, 'w', encoding='utf-8') as f:
                    samples = replay_buffer.all_samples(envs, agent=None)
                    samples = [s for s in samples if not replay_buffer_copy.contain(s.traj)]
                    f.write(show_samples(samples, envs[0].de_vocab, None))

                tf.logging.info('{} finished'.format(self.name))
                return
            i += 1


class OracleActor(Actor):
    '''
    This actor only put oracle examples in the training queue, which means:
    1. This actor do not have a replay buffer
    2. It does not do exploration
    '''

    def __init__(
            self, name, actor_id, shard_ids, ckpt_queue, train_queue, eval_queue, replay_queue):
        Actor.__init__(self, name, actor_id, shard_ids, ckpt_queue, train_queue, eval_queue, replay_queue)
        tf.logging.info('actor_{} is oracle actor'.format(actor_id))

    def run(self):
        agent, all_envs = init_experiment(
            [get_train_shard_path(i) for i in self.shard_ids],
            use_gpu=FLAGS.actor_use_gpu,
            gpu_id=str(self.actor_id + FLAGS.actor_gpu_start_id))
        graph = agent.model.graph
        current_ckpt = get_init_model_path()

        # obtain the oracle of the examples and delete the examples that can not obtain oracle
        envs, env_trajs = get_env_trajs(all_envs)

        # build a dict to store the oracle trajs
        env_oracle_trajs_dict = dict()
        for env, env_traj in zip(envs, env_trajs):
            env_oracle_trajs_dict[env.name] = env_traj
        tf.logging.info('Found oracle for {} envs out of total of {} for actor_{}'.format(len(all_envs), len(envs), self.actor_id))

        i = 0
        while True:
            n_train_samples = 0

            n_train_samples += 1

            # Make sure that all the samples from the env batch
            # fits into one batch for training.
            if FLAGS.batch_size < n_train_samples:
                raise ValueError(
                    'One batch have to at least contain samples from one environment.')

            env_batch_size = FLAGS.batch_size / n_train_samples

            env_iterator = data_utils.BatchIterator(
                dict(envs=envs), shuffle=True,
                batch_size=env_batch_size)

            for j, batch_dict in enumerate(env_iterator):
                batch_envs = batch_dict['envs']
                tf.logging.info('=' * 50)
                tf.logging.info('{} iteration {}, batch {}: {} envs'.format(
                    self.name, i, j, len(batch_envs)))

                t1 = time.time()

                # get the oracle samples
                oracle_samples = []
                for batch_env in batch_envs:
                    oracle_samples.append(agent_factory.Sample(traj=env_oracle_trajs_dict[batch_env.name], prob=1.0))

                self.eval_queue.put((oracle_samples, len(batch_envs)))
                self.replay_queue.put((oracle_samples, len(batch_envs)))

                assert (FLAGS.fixed_replay_weight >= 0.0 and FLAGS.fixed_replay_weight <= 1.0)


                train_samples = []

                train_samples += oracle_samples

                train_samples = sorted(train_samples, key=lambda x: x.traj.env_name)
                tf.logging.info('{} train samples'.format(len(train_samples)))

                if FLAGS.use_importance_sampling:
                    step_logprobs = agent.compute_step_logprobs(
                        [s.traj for s in train_samples])
                else:
                    step_logprobs = None

                # TODO: the clip_factor may be wrong
                self.train_queue.put((train_samples, step_logprobs, 0.0))
                t2 = time.time()
                tf.logging.info(
                    ('{} sec used preparing and enqueuing samples, {}'
                     ' iteration {}, batch {}: {} envs').format(
                        t2-t1, self.name, i, j, len(batch_envs)))

                t1 = time.time()
                # Wait for a ckpt that still exist or it is the same
                # ckpt (no need to load anything).
                while True:
                    new_ckpt = self.ckpt_queue.get()
                    new_ckpt_file = new_ckpt + '.meta'
                    if new_ckpt == current_ckpt or tf.gfile.Exists(new_ckpt_file):
                        break
                t2 = time.time()
                tf.logging.info('{} sec waiting {} iteration {}, batch {}'.format(
                    t2-t1, self.name, i, j))

                if new_ckpt != current_ckpt:
                    # If the ckpt is not the same, then restore the new
                    # ckpt.
                    tf.logging.info('{} loading ckpt {}'.format(self.name, new_ckpt))
                    t1 = time.time()
                    graph.restore(new_ckpt)
                    t2 = time.time()
                    tf.logging.info('{} sec used {} restoring ckpt {}'.format(
                        t2-t1, self.name, new_ckpt))
                    current_ckpt = new_ckpt

            if agent.model.get_global_step() >= FLAGS.n_steps:
                tf.logging.info('{} finished'.format(self.name))
                return
            i += 1


class ActiveActor(Actor):
    '''
    This actor can actively ask for the human annotation for questions with a scheduler

    Upon initialization, this actor does the following steps
    1. load the best model from the pretraining experiment
    2. evaluate the environments (train set) and got the basic information
        a. use beam search
        b. confidence for each hyp at each step
        c. the attention words for each decoding step
    3. refer to the active scheduler for picking and annotating the examples in the train set
    4. after receiving the annotated envs, actor does the following:
        a. put the annotated example in the buffer
        b. mark this example having oracle and rejecting any mismatch spurious forms from entering the buffer
        c. keep exploring (caveat: do NOT know if this is good or bad)
    '''

    def __init__(self, name, actor_id, shard_ids, ckpt_queue, train_queue, eval_queue, replay_queue, env_annotation_dict):
        Actor.__init__(self, name, actor_id, shard_ids, ckpt_queue, train_queue, eval_queue, replay_queue)
        self.env_annotation_dict = env_annotation_dict
        self.decode_vocab = None

    def save_to_buffer(self, examples, replay_buffer):
        filtered_examples = []

        # for the examples that already obtained annotation, screen out the ones that do not fit
        for example in examples:
            annotation = self.env_annotation_dict.get(example.traj.env_name, None)
            if annotation is not None and len(annotation) > 0:
                explored_program = agent_factory.traj_to_program(example.traj, self.decode_vocab)
                if not annotation.verify_program(explored_program):
                    continue
            filtered_examples.append(example)

        replay_buffer.save(filtered_examples)

    def run(self):
        agent, envs = init_experiment(
            [get_train_shard_path(i) for i in self.shard_ids],
            use_gpu=FLAGS.actor_use_gpu,
            gpu_id=str(self.actor_id + FLAGS.actor_gpu_start_id))
        self.decode_vocab = envs[0].de_vocab

        graph = agent.model.graph
        current_ckpt = get_init_model_path()

        env_dict = dict([(env.name, env) for env in envs])
        replay_buffer = agent_factory.AllGoodReplayBuffer(agent, envs[0].de_vocab)

        # Load saved programs to warm start the replay buffer.
        if FLAGS.load_saved_programs:
            load_programs(
                envs, replay_buffer, FLAGS.saved_program_file)

        if FLAGS.save_replay_buffer_at_end:
            replay_buffer_copy = agent_factory.AllGoodReplayBuffer(de_vocab=envs[0].de_vocab)
            replay_buffer_copy.program_prob_dict = copy.deepcopy(replay_buffer.program_prob_dict)

        # shrink the annotation dict to the envs needed
        small_env_annotation_dict = dict()
        for env in envs:
            annotation = self.env_annotation_dict.get(env.name, None)
            if annotation is not None:
                small_env_annotation_dict[env.name] = annotation
        self.env_annotation_dict = small_env_annotation_dict
        print('Actor %d, total %d envs, %d has been annotated.'
              % (self.actor_id, len(envs), len(self.env_annotation_dict)))

        # get samples from the annotations and put them into the buffer
        env_name_dict = dict([(env.name, env) for env in envs])
        if len(self.env_annotation_dict) > 0:
            annotated_samples = []
            for env_name, annotation in self.env_annotation_dict.items():
                samples_from_annotation = annotation.get_samples(env_name_dict[env_name])
                annotated_samples += samples_from_annotation
            self.save_to_buffer(annotated_samples, replay_buffer)

        i = 0
        while True:
            # Create the logging files.
            if FLAGS.log_samples_every_n_epoch > 0 and i % FLAGS.log_samples_every_n_epoch == 0:
                f_replay = codecs.open(os.path.join(
                    get_experiment_dir(), 'replay_samples_{}_{}.txt'.format(self.name, i)),
                    'w', encoding='utf-8')
                f_policy = codecs.open(os.path.join(
                    get_experiment_dir(), 'policy_samples_{}_{}.txt'.format(self.name, i)),
                    'w', encoding='utf-8')
                f_train = codecs.open(os.path.join(
                    get_experiment_dir(), 'train_samples_{}_{}.txt'.format(self.name, i)),
                    'w', encoding='utf-8')

            n_train_samples = 0
            if FLAGS.use_replay_samples_in_train:
                n_train_samples += FLAGS.n_replay_samples

            if FLAGS.use_policy_samples_in_train and FLAGS.use_nonreplay_samples_in_train:
                raise ValueError(
                    'Cannot use both on-policy samples and nonreplay samples for training!')

            if FLAGS.use_policy_samples_in_train or FLAGS.use_nonreplay_samples_in_train:
                # Note that nonreplay samples are drawn by rejection
                # sampling from on-policy samples.
                n_train_samples += FLAGS.n_policy_samples

            # Make sure that all the samples from the env batch
            # fits into one batch for training.
            if FLAGS.batch_size < n_train_samples:
                raise ValueError(
                    'One batch have to at least contain samples from one environment.')

            env_batch_size = FLAGS.batch_size / n_train_samples

            env_iterator = data_utils.BatchIterator(
                dict(envs=envs), shuffle=True,
                batch_size=env_batch_size)

            for j, batch_dict in enumerate(env_iterator):
                batch_envs = batch_dict['envs']
                tf.logging.info('=' * 50)
                tf.logging.info('{} iteration {}, batch {}: {} envs'.format(
                    self.name, i, j, len(batch_envs)))

                t1 = time.time()
                # Generate samples with cache and save to replay buffer.
                t3 = time.time()
                n_explore = 0
                for _ in xrange(FLAGS.n_explore_samples):
                    explore_samples = agent.generate_samples(
                        batch_envs, n_samples=1, use_cache=FLAGS.use_cache,
                        greedy=FLAGS.greedy_exploration)
                    self.save_to_buffer(explore_samples, replay_buffer)
                    n_explore += len(explore_samples)

                if FLAGS.n_extra_explore_for_hard > 0:
                    hard_envs = [env for env in batch_envs
                                 if not replay_buffer.has_found_solution(env.name)]
                    if hard_envs:
                        for _ in xrange(FLAGS.n_extra_explore_for_hard):
                            explore_samples = agent.generate_samples(
                                hard_envs, n_samples=1, use_cache=FLAGS.use_cache,
                                greedy=FLAGS.greedy_exploration)
                            self.save_to_buffer(explore_samples, replay_buffer)
                            n_explore += len(explore_samples)

                t4 = time.time()
                tf.logging.info('{} sec used generating {} exploration samples.'.format(
                    t4 - t3, n_explore))

                tf.logging.info('{} samples saved in the replay buffer.'.format(
                    replay_buffer.size))

                t3 = time.time()
                replay_samples = replay_buffer.replay(
                    batch_envs, FLAGS.n_replay_samples,
                    use_top_k=FLAGS.use_top_k_replay_samples,
                    agent=None if FLAGS.random_replay_samples else agent,
                    truncate_at_n=FLAGS.truncate_replay_buffer_at_n)
                t4 = time.time()
                tf.logging.info('{} sec used selecting {} replay samples.'.format(
                    t4 - t3, len(replay_samples)))

                t3 = time.time()
                if FLAGS.use_top_k_policy_samples:
                    if FLAGS.n_policy_samples == 1:
                        policy_samples = agent.generate_samples(
                            batch_envs, n_samples=FLAGS.n_policy_samples,
                            greedy=True)
                    else:
                        policy_samples = agent.beam_search(
                            batch_envs, beam_size=FLAGS.n_policy_samples)
                else:
                    policy_samples = agent.generate_samples(
                        batch_envs, n_samples=FLAGS.n_policy_samples,
                        greedy=False)
                t4 = time.time()
                tf.logging.info('{} sec used generating {} on-policy samples'.format(
                    t4-t3, len(policy_samples)))

                t2 = time.time()
                tf.logging.info(
                    ('{} sec used generating replay and on-policy samples,'
                     ' {} iteration {}, batch {}: {} envs').format(
                        t2-t1, self.name, i, j, len(batch_envs)))

                t1 = time.time()
                self.eval_queue.put((policy_samples, len(batch_envs)))
                self.replay_queue.put((replay_samples, len(batch_envs)))

                assert (FLAGS.fixed_replay_weight >= 0.0 and FLAGS.fixed_replay_weight <= 1.0)

                if FLAGS.use_replay_prob_as_weight:
                    new_samples = []
                    for sample in replay_samples:
                        name = sample.traj.env_name
                        if name in replay_buffer.prob_sum_dict:
                            replay_prob = max(
                                replay_buffer.prob_sum_dict[name], FLAGS.min_replay_weight)
                        else:
                            replay_prob = 0.0
                        scale = replay_prob
                        new_samples.append(
                            agent_factory.Sample(
                                traj=sample.traj,
                                prob=sample.prob * scale))
                    replay_samples = new_samples
                else:
                    replay_samples = agent_factory.scale_probs(
                        replay_samples, FLAGS.fixed_replay_weight)

                replay_samples = sorted(
                    replay_samples, key=lambda x: x.traj.env_name)

                policy_samples = sorted(
                    policy_samples, key=lambda x: x.traj.env_name)

                if FLAGS.use_nonreplay_samples_in_train:
                    nonreplay_samples = []
                    for sample in policy_samples:
                        if not replay_buffer.contain(sample.traj):
                            nonreplay_samples.append(sample)

                self.save_to_buffer(policy_samples, replay_buffer)

                def weight_samples(samples):
                    if FLAGS.use_replay_prob_as_weight:
                        new_samples = []
                        for sample in samples:
                            name = sample.traj.env_name
                            if name in replay_buffer.prob_sum_dict:
                                replay_prob = max(
                                    replay_buffer.prob_sum_dict[name],
                                    FLAGS.min_replay_weight)
                            else:
                                replay_prob = 0.0
                            scale = 1.0 - replay_prob
                            new_samples.append(
                                agent_factory.Sample(
                                    traj=sample.traj,
                                    prob=sample.prob * scale))
                    else:
                        new_samples = agent_factory.scale_probs(
                            samples, 1 - FLAGS.fixed_replay_weight)
                    return new_samples

                train_samples = []
                if FLAGS.use_replay_samples_in_train:
                    if FLAGS.use_trainer_prob:
                        replay_samples = [
                            sample._replace(prob=None) for sample in replay_samples]
                    train_samples += replay_samples

                if FLAGS.use_policy_samples_in_train:
                    train_samples += weight_samples(policy_samples)

                if FLAGS.use_nonreplay_samples_in_train:
                    train_samples += weight_samples(nonreplay_samples)

                train_samples = sorted(train_samples, key=lambda x: x.traj.env_name)
                tf.logging.info('{} train samples'.format(len(train_samples)))

                if FLAGS.use_importance_sampling:
                    step_logprobs = agent.compute_step_logprobs(
                        [s.traj for s in train_samples])
                else:
                    step_logprobs = None

                if FLAGS.use_replay_prob_as_weight:
                    n_clip = 0
                    for env in batch_envs:
                        name = env.name
                        if (name in replay_buffer.prob_sum_dict and
                                replay_buffer.prob_sum_dict[name] < FLAGS.min_replay_weight):
                            n_clip += 1
                    clip_frac = float(n_clip) / len(batch_envs)
                else:
                    clip_frac = 0.0

                # put all weight on the annotated ones at first, then gradually increase explored examples
                al_scale_factor = min(1.0, (agent.model.get_global_step() - FLAGS.active_start_step) / float(FLAGS.active_scale_steps))
                assert(al_scale_factor >= 0.0 and al_scale_factor <=1.0)
                for i, sample in enumerate(train_samples):
                    if sample.traj.env_name not in self.env_annotation_dict:
                        train_samples[i] = agent_factory.Sample(traj=sample.traj, prob=sample.prob*al_scale_factor)
                    else:
                        annotation = self.env_annotation_dict[sample.traj.env_name]
                        explored_program = agent_factory.traj_to_program(sample.traj, self.decode_vocab)
                        if not annotation.verify_program(explored_program):
                            train_samples[i] = agent_factory.Sample(traj=sample.traj, prob=sample.prob*al_scale_factor)

                self.train_queue.put((train_samples, step_logprobs, clip_frac))
                t2 = time.time()
                tf.logging.info(
                    ('{} sec used preparing and enqueuing samples, {}'
                     ' iteration {}, batch {}: {} envs').format(
                        t2-t1, self.name, i, j, len(batch_envs)))

                t1 = time.time()
                # Wait for a ckpt that still exist or it is the same
                # ckpt (no need to load anything).
                while True:
                    new_ckpt = self.ckpt_queue.get()
                    new_ckpt_file = new_ckpt + '.meta'
                    if new_ckpt == current_ckpt or tf.gfile.Exists(new_ckpt_file):
                        break
                t2 = time.time()
                tf.logging.info('{} sec waiting {} iteration {}, batch {}'.format(
                    t2-t1, self.name, i, j))

                if new_ckpt != current_ckpt:
                    # If the ckpt is not the same, then restore the new
                    # ckpt.
                    tf.logging.info('{} loading ckpt {}'.format(self.name, new_ckpt))
                    t1 = time.time()
                    graph.restore(new_ckpt)
                    t2 = time.time()
                    tf.logging.info('{} sec used {} restoring ckpt {}'.format(
                        t2-t1, self.name, new_ckpt))
                    current_ckpt = new_ckpt

                if FLAGS.log_samples_every_n_epoch > 0 and i % FLAGS.log_samples_every_n_epoch == 0:
                    f_replay.write(show_samples(replay_samples, envs[0].de_vocab, env_dict))
                    f_policy.write(show_samples(policy_samples, envs[0].de_vocab, env_dict))
                    f_train.write(show_samples(train_samples, envs[0].de_vocab, env_dict))

            if FLAGS.log_samples_every_n_epoch > 0 and i % FLAGS.log_samples_every_n_epoch == 0:
                f_replay.close()
                f_policy.close()
                f_train.close()

            if agent.model.get_global_step() >= FLAGS.n_steps:
                if FLAGS.save_replay_buffer_at_end:
                    all_replay = os.path.join(get_experiment_dir(),
                                              'all_replay_samples_{}.txt'.format(self.name))
                with codecs.open(all_replay, 'w', encoding='utf-8') as f:
                    samples = replay_buffer.all_samples(envs, agent=None)
                    samples = [s for s in samples if not replay_buffer_copy.contain(s.traj)]
                    f.write(show_samples(samples, envs[0].de_vocab, None))

                tf.logging.info('{} finished'.format(self.name))
                return
            i += 1

