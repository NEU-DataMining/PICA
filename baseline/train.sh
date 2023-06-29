MODEL_PATH="bert-base-uncased"
DATA_PATH="/datas/zyq/research/emo-llm/datasets/dia_clf/ed/train.jsonl"
OUTPUT_PATH="checkpoints/ed"
wandb_project="emo-bert"
wandb_name="ed-bert"
deepspeed --include localhost:2,3,4,5 bert_clf.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --batch_size 256 \
    --micro_batch_size 64 \
    --num_epochs 100 \
    --learning_rate 1e-5 \
    --val_set_size 0.1 \
    --wandb_project $wandb_project \
    --wandb_run_name $wandb_name \
    --ddp \
    --fp16
