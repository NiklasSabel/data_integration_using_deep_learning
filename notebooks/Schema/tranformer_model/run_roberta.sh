#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --export=NONE
SIZE=$1
BATCH=$2
python transformer_roberta_test.py \
	--model_pretrained_checkpoint /pfs/work7/workspace/scratch/ma_luitheob-tp2021/phd/dev/di-research/reports/contrastive/computers-$SIZE-$BATCH-TinyBERT_General_4L_312D/pytorch_model.bin \ \
    --do_train \
    --do_predict \
    --train_file C:/Users/luisa/OneDrive - bwedu/Documents/BI - Uni Mannheim/Team_Projekt_2021/upload/files/train/small_train_prepped.csv \
	--test_file C:/Users/luisa/OneDrive - bwedu/Documents/BI - Uni Mannheim/Team_Projekt_2021/upload/files/test/test_tables_9000_prepped.csv \
	--tokenizer_name="roberta-base" \
	--grad_checkpoint=True \
    --output_dir C:/Users/luisa/OneDrive - bwedu/Documents/BI - Uni Mannheim/Team_Projekt_2021/upload/output/ \
    --model_name_or_path="roberta-base" \