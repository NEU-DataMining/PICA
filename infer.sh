CUDA_VISIBLE_DEVICES=5 python infer.py \
    --model_name_or_path /datas/huggingface/chatglm2-6b \
    --ptuning_ceckpoint \
    --data_path ./gpt_evaluator/testset.json \
    --batch_size 4 \
    --output_name ./gpt_evaluator/soulchat_result.json
