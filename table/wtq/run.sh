#!/usr/bin/env bash
# basic NSM settings
CONFIG=$1
NAME=$2
USE_NONREPLAY=nouse_nonreplay_samples_in_train
RANDOM_REPLAY=norandom_replay_samples
USE_REPLAY_PROB=nouse_replay_prob_as_weight
N_REPLAY=2
FIXED_REPLAY_WEIGHT=1.0
TOPK_REPLAY=nouse_top_k_replay_samples
USE_TRAINER_PROB=nouse_trainer_prob
TRUNCATE_AT_N=0
BS=50
N_STEPS=25000
OUTPUT=output
LOAD_PROGRAM=load_saved_programs
# about active learning
STARTING_BASELINE=cold
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
DATA_DIR=`cd ../../data/wikitable; pwd`"/"
INPUT_DIR=$DATA_DIR"processed_input/preprocess_14/"
SPLIT_DIR=$INPUT_DIR"data_split_1/"
case $CONFIG in
    active_learning)
        echo active learning based on trained model
        USE_NONREPLAY=use_nonreplay_samples_in_train
        USE_REPLAY_PROB=use_replay_prob_as_weight
        N_REPLAY=5
        START_STEP=0
        if [ $STARTING_BASELINE == "warm" ]; then
            START_STEP=18200
        elif [ $STARTING_BASELINE == "cold" ]; then
            START_STEP=0
        else
            echo option $STARTING_BASELINE for starting baseline is invalid
        fi
        INIT_MODEL_PATH=init_model_path"="$DATA_DIR"output/baseline_"$STARTING_BASELINE"/best_model/model-"$START_STEP
        LOAD_MODEL_FROM_EXPERIMENT=load_model_from_experiment"="$DATA_DIR"output/baseline_"$STARTING_BASELINE
        EXPLORE_EXAMPLES_NUM=5
        AL_START_STEP=$START_STEP
        AL_SCALE_STEPS=20000
        AL_STEPS=50000
        N_STEPS=$((AL_START_STEP+AL_STEPS))
        USE_ACTIVE_LEARNING=use_active_learning
        # We only support AllPicker here, but the examples in the wtq_annotations.bin are picked with correctness-based method
        ACTIVE_PICKER_CLASS=AllPicker
        # This could be either of ( OracleAnnotator: Use the fully-specifed MR as extra supervision
        #                          |SketchAnnotator: Use the MR sketches as extra supervision )
        ACTIVE_ANNOTATOR_CLASS=SketchAnnotator
        # This could be either of (50|100)
        AL_BUDGET_N=100
        ;;
    mapo)
        echo mapo
        USE_NONREPLAY=use_nonreplay_samples_in_train
        USE_REPLAY_PROB=use_replay_prob_as_weight
        N_REPLAY=1
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
       --train_shard_prefix="train_split_shard_90-" \
       --shard_start=0 \
       --shard_end=90 \
       --saved_program_file=$DATA_DIR"processed_input/all_train_saved_programs.json" \
       --embedding_file=$DATA_DIR"raw_input/wikitable_glove_embedding_mat.npy" \
       --vocab_file=$DATA_DIR"raw_input/wikitable_glove_vocab.json" \
       --table_file=$INPUT_DIR"tables.jsonl" \
       --en_vocab_file=$INPUT_DIR"en_vocab_min_count_5.json" \
       --save_every_n=10 \
       --n_explore_samples=$EXPLORE_EXAMPLES_NUM \
       --use_cache \
       --batch_size=$BS \
       --dropout=0.2 \
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
       --entropy_reg_coeff=0.01 \
       --n_steps=$N_STEPS \
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
       --experiment_to_eval=$NAME \
       --output_dir=$DATA_DIR$OUTPUT \
       --experiment_name="eval_"$NAME \
       --eval_file=$INPUT_DIR"test_split.jsonl"
#       --load_saved_programs \
