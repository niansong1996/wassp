#-*- coding: utf-8 -*-
import json
import time
import os
import codecs
import multiprocessing

import tensorflow as tf

from nsm import data_utils
from nsm import agent_factory

from table.utils import get_flags, init_experiment, get_experiment_dir, get_init_model_path, show_samples, get_train_shard_path
from table.utils import load_envs_as_json
from table.actors import Actor, OracleActor, ActiveActor
from table.active_learning import active_learning


FLAGS = get_flags()


def get_saved_experiment_config():
  if FLAGS.experiment_to_eval:
    with open(os.path.join(
        FLAGS.output_dir,
        FLAGS.experiment_to_eval,
        'experiment_config.json'), 'r') as f:
      experiment_config = json.load(f)
      return experiment_config
  else:
    return None


def get_program_shard_path(i):
  return os.path.join(
    FLAGS.saved_programs_dir, FLAGS.program_shard_prefix + str(i) + '.json')


def compress_home_path(path):
  home_folder = os.path.expanduser('~')
  n = len(home_folder)
  if path[:n] == home_folder:
    return '~' + path[n:]
  else:
    return path


def create_experiment_config():
  experiment_config = get_saved_experiment_config()
  if experiment_config:
    FLAGS.embedding_file = os.path.expanduser(experiment_config['embedding_file'])
    FLAGS.vocab_file = os.path.expanduser(experiment_config['vocab_file'])
    FLAGS.en_vocab_file = os.path.expanduser(experiment_config['en_vocab_file'])
    FLAGS.table_file = os.path.expanduser(experiment_config['table_file'])

  experiment_config = {
    'embedding_file': compress_home_path(FLAGS.embedding_file),
    'vocab_file': compress_home_path(FLAGS.vocab_file),
    'en_vocab_file': compress_home_path(FLAGS.en_vocab_file),
    'table_file': compress_home_path(FLAGS.table_file)}

  return experiment_config


def run_experiment():
  print('=' * 100)
  if FLAGS.show_log:
    tf.logging.set_verbosity(tf.logging.INFO)

  experiment_dir = get_experiment_dir()
  if tf.gfile.Exists(experiment_dir):
    tf.gfile.DeleteRecursively(experiment_dir)
  tf.gfile.MkDir(experiment_dir)

  experiment_config = create_experiment_config()
  
  with open(os.path.join(
      get_experiment_dir(), 'experiment_config.json'), 'w') as f:
    json.dump(experiment_config, f)

  ckpt_queue = multiprocessing.Queue()
  train_queue = multiprocessing.Queue()
  eval_queue = multiprocessing.Queue()
  replay_queue = multiprocessing.Queue()

  run_type = 'evaluation' if FLAGS.eval_only else 'experiment'
  print('Start {} {}.'.format(run_type, FLAGS.experiment_name))
  print('The data of this {} is saved in {}.'.format(run_type, experiment_dir))

  if FLAGS.eval_only:
    print('Start evaluating the best model {}.'.format(get_init_model_path()))
  else:
    print('Start distributed training.')

  print('Start evaluator.')
  if FLAGS.eval_on_train:
    print('Evaluating on the training set...')
    evaluator = Evaluator(
      'Evaluator',
      [get_train_shard_path(i) for i in range(FLAGS.shard_start, FLAGS.shard_end)])
  else:
    evaluator = Evaluator(
      'Evaluator',
      [FLAGS.eval_file if FLAGS.eval_only else FLAGS.dev_file])
  evaluator.start()

  if not FLAGS.eval_only:
    actors = []
    actor_shard_dict = dict([(i, []) for i in range(FLAGS.n_actors)])
    for i in xrange(FLAGS.shard_start, FLAGS.shard_end):
      actor_num = i % FLAGS.n_actors
      actor_shard_dict[actor_num].append(i)

    if FLAGS.use_active_learning:
        print('########## use active actor ##########')
        envs = load_envs_as_json([get_train_shard_path(i) for i in range(FLAGS.shard_start, FLAGS.shard_end)])
        al_dict = active_learning(envs, FLAGS.active_picker_class, FLAGS.active_annotator_class, FLAGS.al_budget_n)

    for k in xrange(FLAGS.n_actors):
      name = 'actor_{}'.format(k)

      if FLAGS.use_oracle_examples_in_train:
        actor = OracleActor(name, k, actor_shard_dict[k], ckpt_queue, train_queue, eval_queue, replay_queue)
      elif FLAGS.use_active_learning:
        actor = ActiveActor(name, k, actor_shard_dict[k], ckpt_queue, train_queue, eval_queue, replay_queue, al_dict)
      else:
        actor = Actor(name, k, actor_shard_dict[k], ckpt_queue, train_queue, eval_queue, replay_queue)
      actors.append(actor)
      actor.start()
    print('Start {} actors.'.format(len(actors)))

    print('Start learner.')
    learner = Learner(
      'Learner', [FLAGS.dev_file], ckpt_queue,
      train_queue, eval_queue, replay_queue)
    learner.start()
    print('Use tensorboard to monitor the training progress (see README).')
    for actor in actors:
      actor.join()
    print('All actors finished')
    # Send learner the signal that all the actors have finished.
    train_queue.put(None)
    eval_queue.put(None)
    replay_queue.put(None)
    learner.join()
    print('Learner finished')

  evaluator.join()
  print('Evaluator finished')
  print('=' * 100)




def select_top(samples):
  top_dict = {}
  for sample in samples:
    name = sample.traj.env_name
    prob = sample.prob
    if name not in top_dict or prob > top_dict[name].prob:
      top_dict[name] = sample    
  return agent_factory.normalize_probs(top_dict.values())


def beam_search_eval(agent, envs, writer=None):
    env_batch_size = FLAGS.eval_batch_size
    env_iterator = data_utils.BatchIterator(
      dict(envs=envs), shuffle=False,
      batch_size=env_batch_size)
    dev_samples = []
    dev_samples_in_beam = []
    for j, batch_dict in enumerate(env_iterator):
      t1 = time.time()
      batch_envs = batch_dict['envs']
      tf.logging.info('=' * 50)
      tf.logging.info('eval, batch {}: {} envs'.format(j, len(batch_envs)))
      new_samples_in_beam = agent.beam_search(
        batch_envs, beam_size=FLAGS.eval_beam_size)
      dev_samples_in_beam += new_samples_in_beam
      tf.logging.info('{} samples in beam, batch {}.'.format(
        len(new_samples_in_beam), j))
      t2 = time.time()
      tf.logging.info('{} sec used in evaluator batch {}.'.format(t2 - t1, j))

    # Account for beam search where the beam doesn't
    # contain any examples without error, which will make
    # len(dev_samples) smaller than len(envs).
    dev_samples = select_top(dev_samples_in_beam)
    dev_avg_return, dev_avg_len = agent.evaluate(
      dev_samples, writer=writer, true_n=len(envs))
    tf.logging.info('{} samples in non-empty beam.'.format(len(dev_samples)))
    tf.logging.info('true n is {}'.format(len(envs)))
    tf.logging.info('{} questions in dev set.'.format(len(envs)))
    tf.logging.info('{} dev avg return.'.format(dev_avg_return))
    tf.logging.info('dev: avg return: {}, avg length: {}.'.format(
      dev_avg_return, dev_avg_len))

    return dev_avg_return, dev_samples, dev_samples_in_beam


class Evaluator(multiprocessing.Process):
    
  def __init__(self, name, fns):
    multiprocessing.Process.__init__(self)
    self.name = name
    self.fns = fns

  def run(self):
    agent, envs = init_experiment(self.fns, FLAGS.eval_use_gpu, gpu_id=str(FLAGS.eval_gpu_id))
    for env in envs:
      env.punish_extra_work = False
    graph = agent.model.graph
    dev_writer = tf.summary.FileWriter(os.path.join(
      get_experiment_dir(), FLAGS.tb_log_dir, 'dev'))
    best_dev_avg_return = 0.0
    best_model_path = ''
    best_model_dir = os.path.join(get_experiment_dir(), FLAGS.best_model_dir)
    if not tf.gfile.Exists(best_model_dir):
      tf.gfile.MkDir(best_model_dir)
    i = 0
    current_ckpt = get_init_model_path()
    env_dict = dict([(env.name, env) for env in envs])
    while True:
      t1 = time.time()
      tf.logging.info('dev: iteration {}, evaluating {}.'.format(i, current_ckpt))

      dev_avg_return, dev_samples, dev_samples_in_beam = beam_search_eval(
        agent, envs, writer=dev_writer)
      
      if dev_avg_return > best_dev_avg_return:
        best_model_path = graph.save(
          os.path.join(best_model_dir, 'model'),
          agent.model.get_global_step())
        best_dev_avg_return = dev_avg_return
        tf.logging.info('New best dev avg returns is {}'.format(best_dev_avg_return))
        tf.logging.info('New best model is saved in {}'.format(best_model_path))
        with open(os.path.join(get_experiment_dir(), 'best_model_info.json'), 'w') as f:
          result = {'best_model_path': compress_home_path(best_model_path)}
          if FLAGS.eval_only:
            result['best_eval_avg_return'] = best_dev_avg_return
          else:
            result['best_dev_avg_return'] = best_dev_avg_return
          json.dump(result, f)

      if FLAGS.eval_only:
        # Save the decoding results for further. 
        dev_programs_in_beam_dict = {}
        for sample in dev_samples_in_beam:
          name = sample.traj.env_name
          program = agent_factory.traj_to_program(sample.traj, envs[0].de_vocab)
          answer = sample.traj.answer
          if name in dev_programs_in_beam_dict:
            dev_programs_in_beam_dict[name].append((program, answer, sample.prob))
          else:
            dev_programs_in_beam_dict[name] = [(program, answer, sample.prob)]

        t3 = time.time()
        with open(
            os.path.join(get_experiment_dir(), 'dev_programs_in_beam_{}.json'.format(i)),
            'w') as f:
          json.dump(dev_programs_in_beam_dict, f)
        t4 = time.time()
        tf.logging.info('{} sec used dumping programs in beam in eval iteration {}.'.format(
          t4 - t3, i))

        t3 = time.time()
        with codecs.open(
            os.path.join(
              get_experiment_dir(), 'dev_samples_{}.txt'.format(i)),
            'w', encoding='utf-8') as f:
          for sample in dev_samples:
            f.write(show_samples([sample], envs[0].de_vocab, env_dict))
        t4 = time.time()
        tf.logging.info('{} sec used logging dev samples in eval iteration {}.'.format(
          t4 - t3, i))

      t2 = time.time()
      tf.logging.info('{} sec used in eval iteration {}.'.format(
        t2 - t1, i))

      if FLAGS.eval_only or agent.model.get_global_step() >= FLAGS.n_steps:
        tf.logging.info('{} finished'.format(self.name))
        if FLAGS.eval_only:
          print('Eval average return (accuracy) of the best model is {}'.format(
            best_dev_avg_return))
        else:
          print('Best dev average return (accuracy) is {}'.format(best_dev_avg_return))
          print('Best model is saved in {}'.format(best_model_path))
        return

      # Reload on the latest model.
      new_ckpt = None
      t1 = time.time()
      while new_ckpt is None or new_ckpt == current_ckpt:
        time.sleep(1)
        new_ckpt = tf.train.latest_checkpoint(
          os.path.join(get_experiment_dir(), FLAGS.saved_model_dir))
      t2 = time.time()
      tf.logging.info('{} sec used waiting for new checkpoint in evaluator.'.format(
        t2-t1))
      
      tf.logging.info('lastest ckpt to evaluate is {}.'.format(new_ckpt))
      tf.logging.info('{} loading ckpt {}'.format(self.name, new_ckpt))
      t1 = time.time()
      graph.restore(new_ckpt)
      t2 = time.time()
      tf.logging.info('{} sec used {} loading ckpt {}'.format(
        t2-t1, self.name, new_ckpt))
      current_ckpt = new_ckpt


class Learner(multiprocessing.Process):
    
  def __init__(
      self, name, fns, ckpt_queue,
      train_queue, eval_queue, replay_queue):
    multiprocessing.Process.__init__(self)
    self.ckpt_queue = ckpt_queue
    self.eval_queue = eval_queue
    self.train_queue = train_queue
    self.replay_queue = replay_queue
    self.name = name
    self.save_every_n = FLAGS.save_every_n
    self.fns = fns
      
  def run(self):
    # Writers to record training and replay information.
    train_writer = tf.summary.FileWriter(os.path.join(
      get_experiment_dir(), FLAGS.tb_log_dir, 'train'))
    replay_writer = tf.summary.FileWriter(os.path.join(
      get_experiment_dir(), FLAGS.tb_log_dir, 'replay'))
    saved_model_dir = os.path.join(get_experiment_dir(), FLAGS.saved_model_dir)
    if not tf.gfile.Exists(saved_model_dir):
      tf.gfile.MkDir(saved_model_dir)
    agent, envs = init_experiment(self.fns, FLAGS.train_use_gpu, gpu_id=str(FLAGS.train_gpu_id))
    agent.train_writer = train_writer
    graph = agent.model.graph
    current_ckpt = get_init_model_path()

    i = 0
    n_save = 0
    while True:
      tf.logging.info('Start train step {}'.format(i))
      t1 = time.time()
      train_samples, behaviour_logprobs, clip_frac  = self.train_queue.get()
      eval_samples, eval_true_n = self.eval_queue.get()
      replay_samples, replay_true_n = self.replay_queue.get()
      t2 = time.time()
      tf.logging.info('{} secs used waiting in train step {}.'.format(
        t2-t1, i))
      t1 = time.time()
      n_train_samples = 0
      if FLAGS.use_replay_samples_in_train:
        n_train_samples += FLAGS.n_replay_samples
      if FLAGS.use_policy_samples_in_train and FLAGS.use_nonreplay_samples_in_train:
        raise ValueError(
          'Cannot use both on-policy samples and nonreplay samples for training!')
      if FLAGS.use_policy_samples_in_train:
        n_train_samples += FLAGS.n_policy_samples

      if train_samples:
        if FLAGS.use_trainer_prob:
          train_samples = agent.update_replay_prob(
            train_samples, min_replay_weight=FLAGS.min_replay_weight)
        for _ in xrange(FLAGS.n_opt_step):
          agent.train(
            train_samples,
            parameters=dict(en_rnn_dropout=FLAGS.dropout,rnn_dropout=FLAGS.dropout),
            use_baseline=FLAGS.use_baseline,
            min_prob=FLAGS.min_prob,
            scale=n_train_samples,
            behaviour_logprobs=behaviour_logprobs,
            use_importance_sampling=FLAGS.use_importance_sampling,
            ppo_epsilon=FLAGS.ppo_epsilon,
            de_vocab=envs[0].de_vocab,
            debug=FLAGS.debug)

      avg_return, avg_len = agent.evaluate(
        eval_samples, writer=train_writer, true_n=eval_true_n,
        clip_frac=clip_frac)
      tf.logging.info('train: avg return: {}, avg length: {}.'.format(
        avg_return, avg_len))
      avg_return, avg_len = agent.evaluate(
        replay_samples, writer=replay_writer, true_n=replay_true_n)
      tf.logging.info('replay: avg return: {}, avg length: {}.'.format(avg_return, avg_len))
      t2 = time.time()
      tf.logging.info('{} sec used in training train iteration {}, {} samples.'.format(
        t2-t1, i, len(train_samples)))
      i += 1
      if i % self.save_every_n == 0:
        t1 = time.time()
        current_ckpt = graph.save(
          os.path.join(saved_model_dir, 'model'),
          agent.model.get_global_step())
        t2 = time.time()
        tf.logging.info('{} sec used saving model to {}, train iteration {}.'.format(
          t2-t1, current_ckpt, i))
        self.ckpt_queue.put(current_ckpt)
        if agent.model.get_global_step() >= FLAGS.n_steps:
          t1 = time.time()
          while True:
            train_data = self.train_queue.get()
            _ = self.eval_queue.get()
            _ = self.replay_queue.get()
            self.ckpt_queue.put(current_ckpt)
            # Get the signal that all the actors have
            # finished.
            if train_data is None:
              t2 = time.time()
              tf.logging.info('{} finished, {} sec used waiting for actors'.format(
                self.name, t2-t1))
              return
      else:
        # After training on one set of samples, put one ckpt
        # back so that the ckpt queue is always full.
        self.ckpt_queue.put(current_ckpt)



def main(unused_argv):
  run_experiment()


if __name__ == '__main__':  
    tf.app.run()
