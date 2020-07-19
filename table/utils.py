import re
import os
import json
import time
import sys

import babel
import tensorflow as tf
import numpy as np

from babel import numbers
from wtq import evaluator
from decimal import Decimal

from nsm import agent_factory
from nsm import model_factory
from nsm import data_utils
from nsm import graph_factory
from nsm import computer_factory
from nsm import executor_factory
from nsm import env_factory
from nsm import word_embeddings

# FLAGS
FLAGS = tf.app.flags.FLAGS


# Experiment name
tf.flags.DEFINE_string('output_dir', '', 'output folder.')
tf.flags.DEFINE_string('experiment_name', 'experiment',
                       'All outputs of this experiment is'
                       ' saved under a folder with the same name.')


# Tensorboard logging
tf.app.flags.DEFINE_string(
    'tb_log_dir', 'tb_log', 'Path for saving tensorboard logs.')


# Tensorflow model checkpoint.
tf.app.flags.DEFINE_string(
    'saved_model_dir', 'saved_model', 'Path for saving models.')
tf.app.flags.DEFINE_string(
    'best_model_dir', 'best_model', 'Path for saving best models.')
tf.app.flags.DEFINE_string(
    'init_model_path', '', 'Path for saving best models.')
tf.app.flags.DEFINE_string(
    'noinit_model_path', '', 'Stub for no init model path')
tf.app.flags.DEFINE_string(
    'meta_graph_path', '', 'Path for meta graph.')
tf.app.flags.DEFINE_string('experiment_to_eval', '', '.')
tf.app.flags.DEFINE_string('load_model_from_experiment', '', '.')
tf.app.flags.DEFINE_string('noload_model_from_experiment', '', '.')


# Model
## Computer
tf.app.flags.DEFINE_integer(
    'max_n_mem', 100, 'Max number of memory slots in the "computer".')
tf.app.flags.DEFINE_integer(
    'max_n_exp', 4, 'Max number of expressions allowed in a program.')
tf.app.flags.DEFINE_integer(
    'max_n_valid_indices', 100, 'Max number of valid tokens during decoding.')
tf.app.flags.DEFINE_bool(
    'use_cache', False, 'Use cache to avoid generating the same samples.')
tf.app.flags.DEFINE_string(
    'en_vocab_file', '', '.')
tf.app.flags.DEFINE_string(
    'executor', 'wtq', 'Which executor to use, wtq or wikisql.')


## neural network
tf.app.flags.DEFINE_integer(
    'hidden_size', 100, 'Number of hidden units.')
tf.app.flags.DEFINE_integer(
    'attn_size', 100, 'Size of attention vector.')
tf.app.flags.DEFINE_integer(
    'attn_vec_size', 100, 'Size of the vector parameter for computing attention.')
tf.app.flags.DEFINE_integer(
    'n_layers', 1, 'Number of layers in decoder.')
tf.app.flags.DEFINE_integer(
    'en_n_layers', 1, 'Number of layers in encoder.')
tf.app.flags.DEFINE_integer(
    'en_embedding_size', 100, 'Size of encoder input embedding.')
tf.app.flags.DEFINE_integer(
    'value_embedding_size', 300, 'Size of value embedding for the constants.')
tf.app.flags.DEFINE_bool(
    'en_bidirectional', False, 'Whether to use bidirectional RNN in encoder.')
tf.app.flags.DEFINE_bool(
    'en_attn_on_constants', False, '.')
tf.app.flags.DEFINE_bool(
    'use_pretrained_embeddings', False, 'Whether to use pretrained embeddings.')
tf.app.flags.DEFINE_integer(
    'pretrained_embedding_size', 300, 'Size of pretrained embedding.')


# Features
tf.app.flags.DEFINE_integer(
    'n_de_output_features', 1,
    'Number of features in decoder output softmax.')
tf.app.flags.DEFINE_integer(
    'n_en_input_features', 1,
    'Number of features in encoder inputs.')


# Data
tf.app.flags.DEFINE_string(
    'table_file', '', 'Path to the file of wikitables, a jsonl file.')
tf.app.flags.DEFINE_string(
    'train_file', '', 'Path to the file of training examples, a jsonl file.')
tf.app.flags.DEFINE_string(
    'dev_file', '', 'Path to the file of training examples, a jsonl file.')
tf.app.flags.DEFINE_string(
    'eval_file', '', 'Path to the file of test examples, a jsonl file.')
tf.app.flags.DEFINE_string(
    'embedding_file', '', 'Path to the file of pretrained embeddings, a npy file.')
tf.app.flags.DEFINE_string(
    'vocab_file', '', 'Path to the vocab file for the pretrained embeddings, a json file.')
tf.app.flags.DEFINE_string(
    'train_shard_dir', '', 'Folder containing the sharded training data.')
tf.app.flags.DEFINE_string(
    'train_shard_prefix', '', 'The prefix for the sharded files.')
tf.app.flags.DEFINE_integer(
    'n_train_shard', 90, 'Number of shards in total.')
tf.app.flags.DEFINE_integer(
    'shard_start', 0,
    'Start id of the shard to use.')
tf.app.flags.DEFINE_integer(
    'shard_end', 90, 'End id of the shard to use.')


# Load saved samples.
tf.app.flags.DEFINE_bool(
    'load_saved_programs', False,
    'Whether to use load saved programs from exploration.')
tf.app.flags.DEFINE_bool(
    'noload_saved_programs', True,
    'Whether to use load saved programs from exploration.')
tf.app.flags.DEFINE_string(
    'saved_program_file', '', 'Saved program file.')


# Training
tf.app.flags.DEFINE_integer(
    'n_steps', 100000, 'Maximum number of steps in training.')
tf.app.flags.DEFINE_integer(
    'n_explore_samples', 1, 'Number of exploration samples per env per epoch.')
tf.app.flags.DEFINE_integer(
    'n_extra_explore_for_hard', 0, 'Number of exploration samples for hard envs.')
tf.app.flags.DEFINE_float(
    'learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'max_grad_norm', 5.0, 'Maximum gradient norm.')
tf.app.flags.DEFINE_float(
    'l2_coeff', 0.0, 'l2 regularization coefficient.')
tf.app.flags.DEFINE_float(
    'dropout', 0.0, 'Dropout rate.')
tf.app.flags.DEFINE_integer(
    'batch_size', 10, 'Model batch size.')
tf.app.flags.DEFINE_integer(
    'n_actors', 3, 'Number of actors for generating samples.')
tf.app.flags.DEFINE_integer(
    'save_every_n', -1,
    'Save model to a ckpt every n train steps, -1 means save every epoch.')
tf.app.flags.DEFINE_bool(
    'save_replay_buffer_at_end', True,
    'Whether to save the full replay buffer for each actor at the '
    'end of training or not')
tf.app.flags.DEFINE_integer(
    'log_samples_every_n_epoch', 0,
    'Log samples every n epochs.')
tf.app.flags.DEFINE_bool(
    'greedy_exploration', False,
    'Whether to use a greedy policy when doing systematic exploration.')
tf.app.flags.DEFINE_bool(
    'use_baseline', False,
    'Whether to use baseline during policy gradient.')
tf.app.flags.DEFINE_float(
    'min_prob', 0.0,
    ('Minimum probability of a negative'
     'example for it to be punished to avoid numerical issue.'))
tf.app.flags.DEFINE_float(
    'lm_loss_coeff', 0.0,
    'Weight for lm loss.')
tf.app.flags.DEFINE_float(
    'entropy_reg_coeff', 0.0,
    'Weight for entropy regularization.')
tf.app.flags.DEFINE_string(
    'optimizer', 'adam', '.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9, 'adam beta1 parameter.')
tf.app.flags.DEFINE_bool(
    'sample_other', False, 'Whether to use a greedy policy during training.')
tf.app.flags.DEFINE_bool(
    'use_replay_samples_in_train', False,
    'Whether to use replay samples for training.')
tf.app.flags.DEFINE_bool(
    'random_replay_samples', False,
    'randomly pick a replay samples as ML baseline.')
tf.app.flags.DEFINE_bool(
    'use_policy_samples_in_train', False,
    'Whether to use on-policy samples for training.')
tf.app.flags.DEFINE_bool(
    'use_nonreplay_samples_in_train', False,
    'Whether to use a negative samples for training.')
tf.app.flags.DEFINE_integer(
    'n_replay_samples', 5, 'Number of replay samples drawn.')
tf.app.flags.DEFINE_integer(
    'n_policy_samples', 5, 'Number of on-policy samples drawn.')
tf.app.flags.DEFINE_bool(
    'use_top_k_replay_samples', False,
    ('Whether to use the top k most probable (model probability) replay samples'
     ' or to sample from the replay samples.'))
tf.app.flags.DEFINE_bool(
    'use_top_k_policy_samples', False,
    ('Whether to use the top k most probable (from beam search) samples'
     ' or to sample from the replay samples.'))
tf.app.flags.DEFINE_float(
    'fixed_replay_weight', 0.5, 'Weight for replay samples between 0.0 and 1.0.')
tf.app.flags.DEFINE_bool(
    'use_replay_prob_as_weight', False,
    'Whether or not use replay probability as weight for replay samples.')
tf.app.flags.DEFINE_float(
    'min_replay_weight', 0.1, 'minimum replay weight.')
tf.app.flags.DEFINE_bool(
    'use_importance_sampling', False, '')
tf.app.flags.DEFINE_float(
    'ppo_epsilon', 0.2, '')
tf.app.flags.DEFINE_integer(
    'truncate_replay_buffer_at_n', 0,
    'Whether truncate the replay buffer to the top n highest prob trajs.')
tf.app.flags.DEFINE_bool(
    'use_trainer_prob', False,
    'Whether to supply all the replay buffer for training.')
tf.app.flags.DEFINE_bool(
    'show_log', False,
    'Whether to show logging info.')

# for active learning
tf.app.flags.DEFINE_bool(
    'use_oracle_examples_in_train', False,
    'whether to use actor than only push oracle examples')
tf.app.flags.DEFINE_bool(
    'nouse_oracle_examples_in_train', True,
    'whether to use actor than only push oracle examples')
tf.app.flags.DEFINE_integer(
    'oracle_example_n', 0,
    'duplicate the oracle examples for n times to increase weight')
tf.app.flags.DEFINE_string(
    'active_picker_class', '', 'active learning picker class')
tf.app.flags.DEFINE_string(
    'active_annotator_class', '', 'active learning annotator class')
tf.app.flags.DEFINE_bool(
    'use_active_learning', False,
    'whether to use active learning strategies')
tf.app.flags.DEFINE_bool(
    'nouse_active_learning', True, '')
tf.app.flags.DEFINE_integer(
    'al_budget_n', sys.maxint,
    'number of queries allowed to make')
tf.app.flags.DEFINE_integer(
    'active_start_step', 0,
    'the total steps of the pretrained model')
tf.app.flags.DEFINE_integer(
    'active_scale_steps', 0,
    'the number of steps the scaling of al takes effect')



# Eval
tf.app.flags.DEFINE_integer(
    'eval_beam_size', 5,
    'Beam size when evaluating on development set.')

tf.app.flags.DEFINE_integer(
    'eval_batch_size', 50,
    'Batch size when evaluating on development set.')

tf.app.flags.DEFINE_bool(
    'eval_only', False, 'only run evaluator.')

tf.app.flags.DEFINE_bool(
    'eval_on_train', False, 'evaluate on the train set')

tf.app.flags.DEFINE_bool(
    'debug', False, 'Whether to output debug information.')


# Device placement.
tf.app.flags.DEFINE_bool(
    'train_use_gpu', False, 'Whether to output debug information.')
tf.app.flags.DEFINE_integer(
    'train_gpu_id', 0, 'Id of the gpu used for training.')

tf.app.flags.DEFINE_bool(
    'eval_use_gpu', False, 'Whether to output debug information.')
tf.app.flags.DEFINE_integer(
    'eval_gpu_id', 1, 'Id of the gpu used for eval.')

tf.app.flags.DEFINE_bool(
    'actor_use_gpu', False, 'Whether to output debug information.')
tf.app.flags.DEFINE_integer(
    'actor_gpu_start_id', 0,
    'Id of the gpu for the first actor, gpu for other actors will follow.')


# Testing
tf.app.flags.DEFINE_bool(
    'unittest', False, '.')

tf.app.flags.DEFINE_integer(
    'n_opt_step', 1, 'Number of optimization steps per training batch.')


def get_flags():
    return FLAGS


def average_token_embedding(tks, model, embedding_size=300):
  arrays = []
  for tk in tks:
    if tk in model:
      arrays.append(model[tk])
    else:
      arrays.append(np.zeros(embedding_size))
  return np.average(np.vstack(arrays), axis=0)


def get_embedding_for_constant(value, model, embedding_size=300):
  if isinstance(value, list):
    # Use zero embeddings for values from the question to
    # avoid overfitting
    return np.zeros(embedding_size)
  elif value[:2] == 'r.':
    value_split = value.split('-')
    type_str = value_split[-1]
    type_embedding = average_token_embedding([type_str], model)
    value_split = value_split[:-1]
    value = '-'.join(value_split)
    raw_tks = value[2:].split('_')
    tks = []
    for tk in raw_tks:
      valid_tks = find_tk_in_model(tk, model)
      tks += valid_tks
    val_embedding = average_token_embedding(tks or raw_tks, model)
    return (val_embedding + type_embedding) / 2
  else:
    raise NotImplementedError('Unexpected value: {}'.format(value))


def find_tk_in_model(tk, model):
    special_tk_dict = {'-lrb-': '(', '-rrb-': ')'}
    if tk in model:
        return [tk]
    elif tk in special_tk_dict:
        return [special_tk_dict[tk]]
    elif tk.upper() in model:
        return [tk.upper()]
    elif tk[:1].upper() + tk[1:] in model:
        return [tk[:1].upper() + tk[1:]]
    elif re.search('\\/', tk):
        tks = tk.split('\\\\/')
        if len(tks) == 1:
          return []
        valid_tks = []
        for tk in tks:
            valid_tk = find_tk_in_model(tk, model)
            if valid_tk:
                valid_tks += valid_tk
        return valid_tks
    else:
        return []


# WikiSQL evaluation utility functions.
def wikisql_normalize(val):
  """Normalize the val for wikisql experiments."""
  if (isinstance(val, float) or isinstance(val, int)):
    return float(val)
  elif isinstance(val, Decimal):
    return float(val)
  elif isinstance(val, str) or isinstance(val, unicode):
    try:
      val = float(babel.numbers.parse_decimal(val))
    except (babel.numbers.NumberFormatError, UnicodeEncodeError):
      from table.wikisql.preprocess import to_unicode, normalize
      val = to_unicode(normalize(val.lower()))
    return val
  else:
    return None


def wikisql_process_answer(answer):
  processed_answer = []
  for a in answer:
    normalized_val = wikisql_normalize(a)
    # Ignore None value and normalize the rest, keep the
    # order.
    if normalized_val is not None:
      processed_answer.append(normalized_val)
  return processed_answer


def wikisql_score(prediction, answer):
  prediction = wikisql_process_answer(prediction)
  answer = wikisql_process_answer(answer)
  if prediction == answer:
    return 1.0
  else:
    return 0.0


# WikiTableQuestions evaluation function.
def wtq_score(prediction, answer):
    processed_answer = evaluator.target_values_map(*answer)
    correct = evaluator.check_prediction(
      [unicode(p) for p in prediction], processed_answer)
    if correct:
        return 1.0
    else:
        return 0.0


def collect_traj_for_program(env, program, debug=False):
    env = env.clone()
    env.use_cache = False
    ob = env.start_ob

    for tk in program:
        valid_actions = list(ob[0].valid_indices)
        mapped_action = env.de_vocab.lookup(tk)
        try:
            action = valid_actions.index(mapped_action)
        except Exception as e:
            if debug:
                return None, (env.interpreter.namespace, env.actions, program, mapped_action, valid_actions)
            else:
                return None
        ob, _, _, _ = env.step(action)
    traj = agent_factory.Traj(
        obs=env.obs, actions=env.actions, rewards=env.rewards,
        context=env.get_context(), env_name=env.name, answer=env.interpreter.result)

    if debug:
        return traj, None
    else:
        return traj


def create_agent(graph_config, init_model_path,
                 pretrained_embeddings=None):
    tf.logging.info('Start creating and initializing graph')
    t1 = time.time()
    graph = graph_factory.MemorySeq2seqGraph(graph_config)
    graph.launch(init_model_path=init_model_path)
    t2 = time.time()
    tf.logging.info('{} sec used to create and initialize graph'.format(t2 - t1))

    tf.logging.info('Start creating model and agent')
    t1 = time.time()
    model = model_factory.MemorySeq2seqModel(graph, batch_size=FLAGS.batch_size)

    if pretrained_embeddings is not None:
        model.init_pretrained_embeddings(pretrained_embeddings)
    agent = agent_factory.PGAgent(model)
    t2 = time.time()
    tf.logging.info('{} sec used to create model and agent'.format(t2 - t1))
    return agent


def load_jsonl(fn):
    result = []
    with open(fn, 'r') as f:
        for line in f:
            data = json.loads(line)
            result.append(data)
    return result


def constant_value_embedding_fn(value, embedding_model):
    return get_embedding_for_constant(value, embedding_model, embedding_size=FLAGS.pretrained_embedding_size)


def create_envs(table_dict, data_set, en_vocab, embedding_model):
    all_envs = []
    t1 = time.time()
    if FLAGS.executor == 'wtq':
        score_fn = wtq_score
        process_answer_fn = lambda x: x
        executor_fn = executor_factory.WikiTableExecutor
    elif FLAGS.executor == 'wikisql':
        score_fn = wikisql_score
        process_answer_fn = wikisql_process_answer
        executor_fn = executor_factory.WikiSQLExecutor
    else:
        raise ValueError('Unknown executor {}'.format(FLAGS.executor))

    for i, example in enumerate(data_set):
        if i % 100 == 0:
            tf.logging.info('creating environment #{}'.format(i))
        kg_info = table_dict[example['context']]
        executor = executor_fn(kg_info)
        api = executor.get_api()
        type_hierarchy = api['type_hierarchy']
        func_dict = api['func_dict']
        constant_dict = api['constant_dict']
        interpreter = computer_factory.LispInterpreter(
            type_hierarchy=type_hierarchy,
            max_mem=FLAGS.max_n_mem, max_n_exp=FLAGS.max_n_exp, assisted=True)
        for v in func_dict.values():
            interpreter.add_function(**v)

        interpreter.add_constant(
            value=kg_info['row_ents'], type='entity_list', name='all_rows')

        de_vocab = interpreter.get_vocab()

        constant_value_embedding_fn = lambda x: get_embedding_for_constant(
            x, embedding_model, embedding_size=FLAGS.pretrained_embedding_size)
        env = env_factory.QAProgrammingEnv(
            en_vocab, de_vocab, question_annotation=example,
            answer=process_answer_fn(example['answer']),
            constants=constant_dict.values(),
            interpreter=interpreter,
            constant_value_embedding_fn=constant_value_embedding_fn,
            score_fn=score_fn,
            name=example['id'])
        all_envs.append(env)
    return all_envs


def get_saved_graph_config():
    if FLAGS.experiment_to_eval:
        with open(os.path.join(
                FLAGS.output_dir,
                FLAGS.experiment_to_eval,
                'graph_config.json'), 'r') as f:
            graph_config = json.load(f)
            return graph_config
    elif FLAGS.load_model_from_experiment:
        with open(os.path.join(
                FLAGS.output_dir,
                FLAGS.load_model_from_experiment,
                'graph_config.json'), 'r') as f:
            graph_config = json.load(f)
            return graph_config
    else:
        return None


def load_envs_as_json(fns):
    dataset = []
    for fn in fns:
        dataset += load_jsonl(fn)
    return dataset


def json_to_envs(dataset):
    tables = load_jsonl(FLAGS.table_file)
    table_dict = dict([(table['name'], table) for table in tables])
    embedding_model = word_embeddings.EmbeddingModel(FLAGS.vocab_file, FLAGS.embedding_file)

    with open(FLAGS.en_vocab_file, 'r') as f:
        vocab = json.load(f)
    en_vocab = data_utils.Vocab([])
    en_vocab.load_vocab(vocab)

    # Create environments.
    envs = create_envs(table_dict, dataset, en_vocab, embedding_model)

    return envs


def init_experiment(fns, use_gpu=False, gpu_id='0'):
    dataset = []
    for fn in fns:
        dataset += load_jsonl(fn)
    tf.logging.info('{} examples in dataset.'.format(len(dataset)))
    tables = load_jsonl(FLAGS.table_file)
    table_dict = dict([(table['name'], table) for table in tables])
    tf.logging.info('{} tables.'.format(len(table_dict)))

    # Load pretrained embeddings.
    embedding_model = word_embeddings.EmbeddingModel(
        FLAGS.vocab_file, FLAGS.embedding_file)

    with open(FLAGS.en_vocab_file, 'r') as f:
        vocab = json.load(f)
    en_vocab = data_utils.Vocab([])
    en_vocab.load_vocab(vocab)
    tf.logging.info('{} unique tokens in encoder vocab'.format(
        len(en_vocab.vocab)))
    tf.logging.info('{} examples in the dataset'.format(len(dataset)))

    # Create environments.
    envs = create_envs(table_dict, dataset, en_vocab, embedding_model)
    if FLAGS.unittest:
        envs = envs[:25]
    tf.logging.info('{} environments in total'.format(len(envs)))

    graph_config = get_saved_graph_config()
    if graph_config:
        # If evaluating an saved model, just load its graph
        # config.
        graph_config['use_gpu'] = use_gpu
        graph_config['gpu_id'] = gpu_id
        agent = create_agent(graph_config, get_init_model_path())
    else:
        if FLAGS.use_pretrained_embeddings:
            tf.logging.info('Using pretrained embeddings!')
            pretrained_embeddings = []
            for i in xrange(len(en_vocab.special_tks), en_vocab.size):
                pretrained_embeddings.append(
                    average_token_embedding(
                        find_tk_in_model(
                            en_vocab.lookup(i, reverse=True), embedding_model),
                        embedding_model,
                        embedding_size=FLAGS.pretrained_embedding_size))
            pretrained_embeddings = np.vstack(pretrained_embeddings)
        else:
            pretrained_embeddings = None

        # Model configuration and initialization.
        de_vocab = envs[0].de_vocab
        n_mem = FLAGS.max_n_mem
        n_builtin = de_vocab.size - n_mem
        en_pretrained_vocab_size = en_vocab.size - len(en_vocab.special_tks)

        graph_config = {}
        graph_config['core_config'] = dict(
            max_n_valid_indices=FLAGS.max_n_valid_indices,
            n_mem=n_mem,
            n_builtin=n_builtin,
            use_attn=True,
            attn_size=FLAGS.attn_size,
            attn_vec_size=FLAGS.attn_vec_size,
            input_vocab_size=de_vocab.size,
            en_input_vocab_size=en_vocab.size,
            hidden_size=FLAGS.hidden_size, n_layers=FLAGS.n_layers,
            en_hidden_size=FLAGS.hidden_size, en_n_layers=FLAGS.en_n_layers,
            en_use_embeddings=True,
            en_embedding_size=FLAGS.en_embedding_size,
            value_embedding_size=FLAGS.value_embedding_size,
            en_pretrained_vocab_size=en_pretrained_vocab_size,
            en_pretrained_embedding_size=FLAGS.pretrained_embedding_size,
            add_lm_loss=FLAGS.lm_loss_coeff > 0.0,
            en_bidirectional=FLAGS.en_bidirectional,
            en_attn_on_constants=FLAGS.en_attn_on_constants)

        graph_config['use_gpu'] = use_gpu
        graph_config['gpu_id'] = gpu_id

        graph_config['output_type'] = 'softmax'
        graph_config['output_config'] = dict(
            output_vocab_size=de_vocab.size, use_logits=True)
        aux_loss_list = [('ent_reg', FLAGS.entropy_reg_coeff),]

        if FLAGS.lm_loss_coeff > 0.0:
            aux_loss_list.append(('en_lm_loss', FLAGS.lm_loss_coeff))
        graph_config['train_config'] = dict(
            aux_loss_list=aux_loss_list,
            learning_rate=FLAGS.learning_rate,
            max_grad_norm=FLAGS.max_grad_norm,
            adam_beta1=FLAGS.adam_beta1,
            l2_coeff=FLAGS.l2_coeff,
            optimizer=FLAGS.optimizer, avg_loss_by_n=False)

        agent = create_agent(
            graph_config, get_init_model_path(),
            pretrained_embeddings=pretrained_embeddings)

    with open(os.path.join(get_experiment_dir(), 'graph_config.json'), 'w') as f:
        json.dump(graph_config, f, sort_keys=True, indent=2)

    return agent, envs


def get_train_shard_path(i):
    return os.path.join(
        FLAGS.train_shard_dir, FLAGS.train_shard_prefix + str(i) + '.jsonl')


def get_init_model_path():
    if FLAGS.init_model_path:
        tf.logging.info('Found init model %s' % FLAGS.init_model_path)
        return FLAGS.init_model_path
    elif FLAGS.experiment_to_eval:
        with open(os.path.join(
                FLAGS.output_dir,
                FLAGS.experiment_to_eval,
                'best_model_info.json'), 'r') as f:
            best_model_info = json.load(f)
            best_model_path = os.path.expanduser(
                best_model_info['best_model_path'])
            return best_model_path
    else:
        return ''


def load_programs(envs, replay_buffer, fn):
    if not tf.gfile.Exists(fn):
        return
    with open(fn, 'r') as f:
        program_dict = json.load(f)
    trajs = []
    n = 0
    total_env = 0
    n_found = 0
    for env in envs:
        total_env += 1
        found = False
        if env.name in program_dict:
            program_str_list = program_dict[env.name]
            n += len(program_str_list)
            env.cache._set = set(program_str_list)
            for program_str in program_str_list:
                program = program_str.split()
                traj = collect_traj_for_program(env, program)
                if traj is not None:
                    trajs.append(traj)
                    if not found:
                        found = True
                        n_found += 1
    tf.logging.info('@' * 100)
    tf.logging.info('loading programs from file {}'.format(fn))
    tf.logging.info('at least 1 solution found fraction: {}'.format(
        float(n_found) / total_env))
    replay_buffer.save_trajs(trajs)
    n_trajs_buffer = 0
    for k, v in replay_buffer._buffer.iteritems():
        n_trajs_buffer += len(v)
    tf.logging.info('{} programs in the file'.format(n))
    tf.logging.info('{} programs extracted'.format(len(trajs)))
    tf.logging.info('{} programs in the buffer'.format(n_trajs_buffer))
    tf.logging.info('@' * 100)


def get_experiment_dir():
    experiment_dir = os.path.join(FLAGS.output_dir, FLAGS.experiment_name)
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MkDir(FLAGS.output_dir)
    if not tf.gfile.IsDirectory(experiment_dir):
        tf.gfile.MkDir(experiment_dir)
    return experiment_dir


def unpack_program(program_str, env):
    ns = env.interpreter.namespace
    processed_program = []
    for tk in program_str.split():
        if tk[:1] == 'v' and tk in ns:
            processed_program.append(unicode(ns[tk]['value']))
        else:
            processed_program.append(tk)
    return ' '.join(processed_program)


def show_samples(samples, de_vocab, env_dict=None):
    string = ''
    for sample in samples:
        traj = sample.traj
        actions = traj.actions
        obs = traj.obs
        pred_answer = traj.answer
        string += u'\n'
        env_name = traj.env_name
        string += u'env {}\n'.format(env_name)
        if env_dict is not None:
            string += u'question: {}\n'.format(env_dict[env_name].question_annotation['question'])
            string += u'answer: {}\n'.format(env_dict[env_name].question_annotation['answer'])
        tokens = []
        program = []
        for t, (a, ob) in enumerate(zip(actions, obs)):
            ob = ob[0]
            valid_tokens = de_vocab.lookup(ob.valid_indices, reverse=True)
            token = valid_tokens[a]
            program.append(token)
        program_str = ' '.join(program)
        if env_dict:
            program_str = unpack_program(program_str, env_dict[env_name])
        string += u'program: {}\n'.format(program_str)
        string += u'prediction: {}\n'.format(pred_answer)
        string += u'return: {}\n'.format(sum(traj.rewards))
        string += u'prob is {}\n'.format(sample.prob)
    return string


def get_statements(program):
    statements = []

    line = []
    for token in program:
        if token == '<END>':
            break
        else:
            line.append(token)
            if token == ')':
                statements.append(line)
                line = []

    return statements


def get_sketch(program):
    statements = get_statements(program)
    sketch = [statement[1] for statement in statements]
    return sketch
