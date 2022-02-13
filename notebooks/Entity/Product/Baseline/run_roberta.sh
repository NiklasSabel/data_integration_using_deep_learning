#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --export=NONE

SIZE=$1
BATCH=$2
python transformer_roberta_debugged.py \
    --do_train \
    --do_predict \
    --num_train_epochs=100\
    --train_file /pfs/work7/workspace/scratch/ma_nsabel-teamprojekt/data_integration_using_deep_learning/src/data/product/train_test_split/output_unfiltered_tables/large/after_manual_checking/baselines/df_train.csv \
    --validation_file /pfs/work7/workspace/scratch/ma_nsabel-teamprojekt/data_integration_using_deep_learning/src/data/product/train_test_split/output_unfiltered_tables/large/after_manual_checking/baselines/df_val.csv \
    --test_file /pfs/work7/workspace/scratch/ma_nsabel-teamprojekt/data_integration_using_deep_learning/src/data/product/train_test_split/output_unfiltered_tables/large/after_manual_checking/baselines/df_test.csv \
    --tokenizer_name="huawei-noah/TinyBERT_General_4L_312D" \
    --gradient_accumulation_steps=2 \
    --logging_steps=3 \
    --warmup_steps=10 \
    --load_best_model_at_end=True\
    --metric_for_best_model="eval_accuracy"\
    --evaluation_strategy="epoch" \
    --save_strategy="epoch" \
    --output_dir /pfs/work7/workspace/scratch/ma_nsabel-teamprojekt/data_integration_using_deep_learning/notebooks/Entity/Product/Baseline/output_tinybert/ \
    --model_name_or_path="huawei-noah/TinyBERT_General_4L_312D" \