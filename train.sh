TOT_CUDA="0,1,2,3,4,5"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="11451"

MODEL_PATH="llama-7b"
DATA_PATH="./datasets/train_12k.jsonl"
TEST_SIZE=500
wandb_name="llama-7b-emo-llm-12k"
wandb_project="emo-llm"

use_deepspeed=1
if [ ${use_deepspeed} == 0 ]
then
    CUDA_VISIBLE_DEVICES=${TOT_CUDA} deepspeed --master_port=$PORT finetune_deepspeed.py \
        --base_model $MODEL_PATH \
        --data_path  $DATA_PATH \
        --output_dir ./checkpoints/$wandb_name \
        --batch_size 128 \
        --micro_batch_size 4 \
        --num_epochs 10 \
        --val_set_size $TEST_SIZE \
        --learning_rate 3e-4 \
        --cutoff_len 1024 \
        --lora_r 8 \
        --lora_alpha 16 \
        --wandb_project $wandb_project \
        --wandb_run_name $wandb_name 
else
    deepspeed --include localhost:0,1,2,3,4,5 finetune.py \
        --base_model $MODEL_PATH \
        --data_path  $DATA_PATH \
        --output_dir ./checkpoints/$wandb_name \
        --batch_size 128 \
        --micro_batch_size 4 \
        --num_epochs 10 \
        --val_set_size $TEST_SIZE \
        --learning_rate 3e-4 \
        --cutoff_len 1024 \
        --lora_r 8 \
        --lora_alpha 16 \
        --wandb_project $wandb_project \
        --wandb_run_name $wandb_name 
fi