CUDA_VISIBLE_DEVICES=4 python infer.py \
    --model_path /datas/huggingface/llama/llama-7b \
    --lora_path ./checkpoints/llama-7b-emo-llm-12k \
    --save_path llama-7b-emo-llm-12k \
    --filename dia_clf-test.jsonl \
    --batch_size 6 \
    --temperature 0.7