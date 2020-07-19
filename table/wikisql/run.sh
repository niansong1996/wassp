#!/usr/bin/env bash
CONFIG=$1
NAME=$2
USE_NONREPLAY=nouse_nonreplay_samples_in_train
RANDOM_REPLAY=norandom_replay_samples
USE_REPLAY_PROB=nouse_replay_prob_as_weight
FIXED_REPLAY_WEIGHT=1.0
TOPK_REPLAY=nouse_top_k_replay_samples
USE_TRAINER_PROB=nouse_trainer_prob
TRUNCATE_AT_N=5
N_STEPS=15000
OUTPUT=output
LOAD_PROGRAM=load_saved_programs
# about active learning
USE_ORACLE_EXAMPLES=nouse_oracle_examples_in_train
ORACLE_EXAMPLES_NUM=0
EXPLORE_EXAMPLES_NUM=1
EXPLORE_HARD_EXAMPLES_NUM=9
INIT_MODEL_PATH=noinit_model_path
LOAD_MODEL_FROM_EXPERIMENT=noload_model_from_experiment
USE_ACTIVE_LEARNING=nouse_active_learning
ACTIVE_PICKER_CLASS=none
ACTIVE_ANNOTATOR_CLASS=none
AL_BUDGET_N=100000
AL_START_STEP=0
AL_SCALE_STEPS=0
# declare some directory locations
DATA_DIR=`cd ../../data/wikisql; pwd`"/"
INPUT_DIR=$DATA_DIR"processed_input/preprocess_4/"
SPLIT_DIR=$INPUT_DIR
case $CONFIG in
    mapo)
        echo mapo
        N_REPLAY=5
        TOPK_REPLAY=use_top_k_replay_samples
        USE_TRAINER_PROB=use_trainer_prob
        USE_NONREPLAY=use_nonreplay_samples_in_train
        USE_REPLAY_PROB=use_replay_prob_as_weight
        BS=750
        ;;
    active_learning)
        echo active learning based on trained model
        N_REPLAY=5
        TOPK_REPLAY=use_top_k_replay_samples
        USE_TRAINER_PROB=use_trainer_prob
        USE_NONREPLAY=use_nonreplay_samples_in_train
        USE_REPLAY_PROB=use_replay_prob_as_weight
        LOAD_PROGRAM=noload_saved_programs
        EXPLORE_EXAMPLES_NUM=1
        EXPLORE_HARD_EXAMPLES_NUM=0
        INIT_MODEL_PATH=init_model_path"="$DATA_DIR"output/imp_baseline/best_model/model-14920"
        LOAD_MODEL_FROM_EXPERIMENT=load_model_from_experiment"="$DATA_DIR"output/imp_baseline"
        AL_START_STEP=14920
        AL_SCALE_STEPS=10000
        AL_STEPS=15000
        N_STEPS=$((AL_START_STEP+AL_STEPS))
        USE_ACTIVE_LEARNING=use_active_learning
        # This could be either of ( AllPicker: Random baseline
        #                          |FailedPicker: Corretness-based method
        #                          |ConfidencePicker: Uncertainty-based method
        #                          |FailedConfidencePicker: Correctness+Uncertainty
        #                          |WordPicker: Coverage-based method (failed word coverage)
        #                          |ClusterPicker: Coverage-based method (Clustering) )
        ACTIVE_PICKER_CLASS=FailedPicker
        # This could be either of ( OracleAnnotator: Use the fully-specifed MR as extra supervision
        #                          |SketchAnnotator: Use the MR sketches as extra supervision )
        ACTIVE_ANNOTATOR_CLASS=OracleAnnotator
        AL_BUDGET_N=1000
        BS=750
        ;;
    *)
        echo "Usage: $0 (mapo|active_learning) experiment_name"
        exit 1
        ;;
esac
python ../experiment.py \
       --output_dir=$DATA_DIR$OUTPUT \
       --experiment_name=$NAME \
       --n_actors=30 \
       --dev_file=$SPLIT_DIR"dev_split.jsonl" \
       --train_shard_dir=$SPLIT_DIR \
       --train_shard_prefix="train_split_shard_30-" \
       --shard_start=0 \
       --shard_end=30 \
       --embedding_file=$DATA_DIR"raw_input/wikisql_glove_embedding_mat.npy" \
       --vocab_file=$DATA_DIR"raw_input/wikisql_glove_vocab.json" \
       --table_file=$INPUT_DIR"tables.jsonl" \
       --en_vocab_file=$INPUT_DIR"en_vocab_min_count_5.json" \
       --$LOAD_PROGRAM \
       --saved_program_file=$DATA_DIR"processed_input/all_train_saved_programs-1k_5.json" \
       --save_every_n=10 \
       --save_replay_buffer_at_end \
       --n_explore_samples=$EXPLORE_EXAMPLES_NUM \
       --n_extra_explore_for_hard=$EXPLORE_HARD_EXAMPLES_NUM \
       --use_cache \
       --batch_size=$BS \
       --dropout=0.0 \
       --hidden_size=200 \
       --attn_size=200 \
       --attn_vec_size=200 \
       --en_embedding_size=200 \
       --en_bidirectional \
       --n_layers=2 \
       --en_n_layers=2 \
       --use_pretrained_embeddings \
       --pretrained_embedding_size=300 \
       --value_embedding_size=300 \
       --learning_rate=0.001 \
       --n_policy_samples=1 \
       --n_replay_samples=$N_REPLAY \
       --use_replay_samples_in_train \
       --$USE_NONREPLAY \
       --$USE_REPLAY_PROB \
       --$TOPK_REPLAY \
       --fixed_replay_weight=$FIXED_REPLAY_WEIGHT \
       --$RANDOM_REPLAY \
       --min_replay_weight=0.1 \
       --truncate_replay_buffer_at_n=$TRUNCATE_AT_N \
       --train_use_gpu \
       --train_gpu_id=0 \
       --eval_use_gpu \
       --eval_gpu_id=1 \
       --max_n_mem=60 \
       --max_n_valid_indices=60 \
       --max_n_exp=4 \
       --eval_beam_size=5 \
       --executor="wikisql" \
       --n_steps=$N_STEPS \
       --$USE_ORACLE_EXAMPLES \
       --oracle_example_n=$ORACLE_EXAMPLES_NUM \
       --$INIT_MODEL_PATH \
       --$LOAD_MODEL_FROM_EXPERIMENT \
       --$USE_ACTIVE_LEARNING \
       --al_budget_n=$AL_BUDGET_N \
       --active_picker_class=$ACTIVE_PICKER_CLASS \
       --active_annotator_class=$ACTIVE_ANNOTATOR_CLASS \
       --active_start_step=$AL_START_STEP \
       --active_scale_steps=$AL_SCALE_STEPS \
       --show_log
python ../experiment.py \
       --eval_only \
       --eval_use_gpu \
       --eval_gpu_id=0 \
       --experiment_name="eval_"$NAME \
       --experiment_to_eval=$NAME \
       --output_dir=$DATA_DIR$OUTPUT \
       --eval_file=$INPUT_DIR"test_split.jsonl" \
       --executor="wikisql"
#--load_saved_programs \
#--saved_program_file=$DATA_DIR"processed_inpu/all_train_saved_programs-1k_5.json" \
