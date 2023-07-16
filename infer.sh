CUDA_VISIBLE_DEVICES=1 python infer.py \
    --data_path ./gpt_evaluator/testset.json \
    --batch_size 4 \
    --output_name ./gpt_evaluator/chatglm2_result.json
