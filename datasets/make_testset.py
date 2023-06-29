# %%
import json
import os
from typing import List

import tiktoken
from prompter import Prompter
from tqdm.auto import tqdm

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

# load instruction config file 
config = json.load(open('dataset_config.json', 'r'))
instruction_config = json.load(open('instruction_config.json', 'r'))
# load gpt encoder
encoder = tiktoken.get_encoding('cl100k_base')
max_token_length: int = 1000

def process_lengthy_input(input: str, instructions: List):
    remain_token_length = max_token_length

    instructions = " ".join(instructions)
    remain_token_length -= len(encoder.encode(instructions))
    
    # input = " ".join(input.split())
    input_encode = encoder.encode(input)
    if len(input_encode) > remain_token_length:
        input_encode = input_encode[-remain_token_length:]
        input = encoder.decode(input_encode)

    return input

def txt_clf_process(example, prompter: Prompter):
    example['name'] = example['dataset'] + '-classification'
    
    instruction = instruction_config[example['name']]
    input = example['input']

    input = process_lengthy_input(input = input, instructions=[instruction])

    example['prompt'] = prompter.txt_clf_generate(
        instruction=instruction,
        input = input,
        emotion = '',
    )
    return example

def dialog_process(history, utterance, ai_last=False, pre_ai = "[ai]", pre_user = "[user]") -> str:
    history += [utterance]
    if ai_last and len(history) % 2 == 1:
        history.pop(0)
    new_history = []
    for i, item in enumerate(history):
        if i % 2 == 0:
            new_history.append(pre_user + ': ' + item)
        else:
            new_history.append(pre_ai + ': ' + item)
    if ai_last:
        new_history.pop()

    return "\n".join(new_history)

def dia_clf_process(example, prompter: Prompter):
    example['name'] = example['dataset'] + '-classification'
    instruction = instruction_config[example['name']]

    dialog = dialog_process(
        history   = example['history'],
        utterance = example['utterance']
    )

    dialog = process_lengthy_input(input = dialog, instructions=[instruction])

    example['prompt'] = prompter.dia_clf_generate(
        instruction = instruction,
        dialog      = dialog,
        emotion     = ''
    )

    return example

def er_abs_process(example, prompter: Prompter):
    example['name'] = example['dataset'] + '-abstract'
    instruction = instruction_config[example['name']]

    input = process_lengthy_input(input = example['input'], instructions=[ instruction, example['emotion'] ])

    example['prompt'] = prompter.er_abs_generate(
        instruction = instruction,
        input       = input,
        emotion     = example['emotion'],
        cause       = ''
    )
    # convert cause to output
    example['output'] = example['cause']

    return example


def dia_inf_process(example, prompter: Prompter):
    example['name'] = example['dataset'] + '-inference'

    if example['dataset'] in ['dailydialog', 'esconv', 'mastodon', 'ESConv', 'ESConv-act']:
        infer_type = 'act'
    elif example['dataset'] in ['empatheticdialogues', 'ed', 'ESConv-situation']:
        infer_type = 'situation'
    elif example['dataset'] in ['personaminedit', 'PersonaMinEdit']:
        infer_type = 'personas'
    elif example['dataset'] in ['mixed-emotion']:
        infer_type = 'emotion'

    instruction = instruction_config[example['name']]

    dialog = dialog_process(
        history   = example['history'],
        utterance = example['utterance']
    )

    dialog = process_lengthy_input(input = dialog, instructions=[instruction, infer_type])

    example['prompt'] = prompter.dia_inf_generate(
        instruction = instruction,
        dialog      = dialog,
        type        = infer_type,
        output      = ''
    )

    return example

def dia_gen_process(example, prompter: Prompter):
    example['name'] = example['dataset'] + '-generation'
    
    dialog = dialog_process(
        history   = example['history'],
        utterance = example['utterance'],
        ai_last   = True
    )
    instruction = "The following is a conversation between an AI assistant called [ai] and a human user called [user]. The [ai] should comply with the following control information. If control information is none, please omit it."

    dialog = process_lengthy_input(input = dialog, instructions=[instruction, example['personas'], example['emotion'], example['situation'], example['act']])

    example['prompt'] = prompter.dia_gen_generate(
        dialog    = dialog,
        personas  = example['personas'],
        emotion   = example['emotion'],
        situation = example['situation'],
        act       = example['act']
    )

    example['output'] = example['utterance']

    return example


seed          = 42
do_sample     = False
sample_number = 5000

task_type_list = ['dia_inf', 'dia_gen', 'dia_clf', 'txt_clf', 'er_abs']
# task_type_list = ['dia_gen']

# make test data
for task_type in tqdm(task_type_list):
    file_lists = list(config[task_type].keys())

    datasets = []

    prompter = Prompter(template_name=task_type)

    for file in tqdm(file_lists):
        data_path = os.path.join(task_type, file, 'test.jsonl')

        temp_set = load_dataset(
            'json',
            data_files = data_path,
            split      = 'train'
        )

        # sample
        if do_sample:
            temp_set = temp_set.shuffle(seed=seed).select(
                range(min(len(temp_set), sample_number))
            )

        if task_type == 'txt_clf':
            temp_set = temp_set.map(lambda x: txt_clf_process(x, prompter), num_proc=8)
        elif task_type == 'dia_clf':
            temp_set = temp_set.map(lambda x: dia_clf_process(x, prompter), num_proc=8)
        elif task_type == 'er_abs':
            temp_set = temp_set.map(lambda x: er_abs_process(x, prompter), num_proc=8)
        elif task_type == 'dia_inf':
            temp_set = temp_set.map(lambda x: dia_inf_process(x, prompter), num_proc=8)
        elif task_type == 'dia_gen':
            temp_set = temp_set.filter(
                lambda x: len(x['history']) % 2 == 1, num_proc=8
            )
            temp_set = temp_set.map(lambda x: dia_gen_process(x, prompter), num_proc=8)
            
        datasets.append(temp_set)
        
    dataset = concatenate_datasets(datasets)
    stable_list = ['name', 'prompt', 'output']
    remove_list = list(
        set(dataset.column_names) - set(stable_list)
    )
    dataset = dataset.remove_columns(remove_list)

    if do_sample:
        dataset.to_json(f'./testset/{task_type}-test-sample.jsonl')
    else:
        dataset.to_json(f'./testset/{task_type}-test.jsonl')
# %%
