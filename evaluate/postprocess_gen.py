import json
from datasets import load_dataset
import datasets
import fire
import os
import re

def process(example):
    gen = example['result']
    prefix1 = "\n[ai]: "
    prefix2 = "package com.example.android.sunshine.app;"
    if len(gen) > 0:
        gen = gen.replace(prefix1, "").split("\n")[0]
        gen = gen.replace(prefix2, "")

    example['gen'] = gen
    prompt = example['prompt']
    pattern = r"## emotion:\n(.*)\n\n## situation"
    match = re.search(pattern, prompt)

    if match:
        example['emotion'] = match.group(1)
    else:
        example['emotion'] = "None"
    
    example['target'] = example['label']
    example['context'] = prompt.split("\n[user]: ")[-1]

    return example

def run(
    root_dir  : str,
    file_name : str,
    output_dir: str,
    key       : str,
):
    dataset = load_dataset(
        'json',
        data_files = os.path.join(root_dir, file_name),
        split      = 'train'
    )
    dataset = dataset.filter(lambda x: x['name'] == key)
    print(dataset)
    dataset = dataset.map(process, num_proc=8)
    dataset = dataset.remove_columns(
        ["prompt", "label", "name", "predict", "result"]
    )
    dataset.to_json(
        os.path.join(output_dir, f'{key}.json')
    )
    # output format
    ## context, target, emotion, gen

if __name__ == '__main__':
    # fire.Fire(run)

    keys = [
        "empatheticdialogues-generation",
        "edos-generation"
    ]
    for key in keys:
        run(
            root_dir='./llama-7b-emo-llm-12k',
            file_name='dia_gen.json',
            output_dir="/datas/zyq/research/emo-llm/evaluate/EmpGPT-3/path/prompt_result",
            key=key
        )