import argparse
import json
import os

import datasets
import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, set_seed
from transformers.pipelines.pt_utils import KeyDataset


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def build_prompt(query, history=None):
    if history is None:
        history = []
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, parse_text(response))
    prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
    return prompt

def preprocess(example):
    query = parse_text(example['prompt'])
    history = example['history'] if len(example['history']) > 0 else None

    prompt = build_prompt(query=query, history=history)
    example['query'] = prompt

    return example

def collate_fn(batch):
    queries = [item['query'] for item in batch]

    tokenized_inputs = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors='pt'
    )

    return {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
    }

parser = argparse.ArgumentParser(description='infer paramters')


parser.add_argument('--model_name_or_path', type=str, default='/datas/huggingface/chatglm2-6b')
parser.add_argument('--ptuning_checkpoint', type=str, default=None)
parser.add_argument('--pre_seq_len', type=int, default=128)
parser.add_argument('--data_path', type=str)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--output_name', type=str)

args = parser.parse_args()

set_seed(42)

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)

config.pre_seq_len = args.pre_seq_len

if args.ptuning_checkpoint is not None:
    print(f"Loading prefix_encoder weight from {args.ptuning_checkpoint}")
    model = AutoModel.from_pretrained(args.model_name_or_path, config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(args.ptuning_checkpoint, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
else:
    model = AutoModel.from_pretrained(args.model_name_or_path, config=config, trust_remote_code=True)
model.cuda()
if args.pre_seq_len is not None:
    model.transformer.prefix_encoder.float()


dataset = load_dataset(
    'json',
    data_files=args.data_path,
    split='train'
)

dataset = dataset.map(preprocess, num_proc=8)

print(dataset['query'][2])

results = []

with torch.no_grad():
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    for batch in tqdm(dataloader):
        inputs = {
            "input_ids": batch["input_ids"].to('cuda'),
            "attention_mask": batch["attention_mask"].to('cuda')
        }

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=256)

        for idx, output in enumerate(outputs):
            output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
            response = parse_text(tokenizer.decode(output))
            
            with open('temp.txt', 'a', encoding='utf-8') as w:
                w.write(str(response) + '\n')

            results.append(
                parse_text(response)
            )


fp = open(f'{args.output_name}', 'a', encoding='utf-8')
for data, result in zip(dataset, results):
    data["result"] = result

    fp.write(
        json.dumps(data, ensure_ascii=False) + '\n'
    )





