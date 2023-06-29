# %%
import json
from datasets import load_dataset, concatenate_datasets
import os
from prompter import Prompter

config = json.load(open('dataset_config.json', 'r'))
instruction_config = json.load(open('instruction_config.json', 'r'))

config, instruction_config
# %%
def txt_clf_process(example, prompter: Prompter):
    example['name'] = example['dataset'] + '-classification'
    
    instruction = instruction_config[example['name']]
    input = example['input']
    output = example['output']

    example['prompt'] = prompter.txt_clf_generate(
        instruction = instruction,
        input       = input,
        emotion     = output,
    )

    return example

def txt_clf_sample_dataset(dataset, sample_nums, seed, prompter, do_sample=True):
    if do_sample:
        label_map = list(set(dataset['output']))
        dataset_list = []
        for label in label_map:
            temp_set = dataset.filter(lambda x: x['output'] == label, num_proc=8)
            temp_set = temp_set.shuffle(seed=seed)
            temp_set = temp_set.select(range(min(sample_nums, len(temp_set))))
            dataset_list.append(temp_set)
        
        dataset = concatenate_datasets(dataset_list)

    dataset = dataset.map(lambda x: txt_clf_process(x, prompter), num_proc=8)
    return dataset
# %%
def dialog_process(history, utterance, ai_last=False, pre_ai="[ai]", pre_user="[user]"):
    history += [utterance]
    if ai_last and len(history) % 2 == 1:
        history.pop(0)

    new_list = []
    for i, item in enumerate(history):
        if i % 2 == 0:
            new_list.append(pre_user+': '+ item)
        else:
            new_list.append(pre_ai+': ' + item)
    
    return "\n".join(new_list)

def dia_clf_process(example, prompter: Prompter):
    example['name'] = example['dataset'] + '-classification'
    instruction = instruction_config[example['name']]
    # 处理对话
    dialog = dialog_process(
        history   = example['history'],
        utterance = example['utterance'],
    )
    example['prompt'] = prompter.dia_clf_generate(
        instruction = instruction,
        dialog      = dialog,
        emotion     = example['output']
    )
    
    return example

def dia_clf_sample_dataset(dataset, sample_nums, seed, prompter, do_sample=True):
    if do_sample:
        label_map = list(set(dataset['output']))
        dataset_list = []
        for label in label_map:
            temp_set = dataset.filter(lambda x: x['output'] == label, num_proc=8)
            temp_set = temp_set.shuffle(seed=seed)
            temp_set = temp_set.select(range(min(sample_nums, len(temp_set))))
            dataset_list.append(temp_set)
    
        dataset = concatenate_datasets(dataset_list)

    dataset = dataset.map(lambda x: dia_clf_process(x, prompter), num_proc=8)

    return dataset
# %%
def er_abs_process(example, prompter:Prompter):
    example['name'] = example['dataset'] + '-abstract'

    instruction = instruction_config[example['name']]

    example['prompt'] = prompter.er_abs_generate(
        instruction = instruction,
        input       = example['input'],
        emotion     = example['emotion'],
        cause       = example['cause']
    )

    return example

def er_abs_sample_dataset(dataset, sample_nums, seed, prompter, do_sample=True):
    if do_sample:
        dataset = dataset.shuffle(seed)   
        dataset = dataset.select(range(min(sample_nums, len(dataset))))

    dataset = dataset.map(lambda x: er_abs_process(x, prompter), num_proc = 8)

    return dataset
# %%
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
    
    dialog = dialog_process(
        history   = example['history'],
        utterance = example['utterance']
    )

    instruction = instruction_config[example['name']]
    example['prompt'] = prompter.dia_inf_generate(
        instruction = instruction,
        dialog      = dialog,
        type        = infer_type,
        output      = example['output']
    )
    
    return example

def dia_inf_sample_dataset(dataset, sample_nums, seed, prompter, do_sample=True):
    if do_sample:
        dataset = dataset.shuffle(seed)
        dataset = dataset.select(
            range(min(sample_nums, len(dataset)))
        )

    dataset = dataset.map(lambda x: dia_inf_process(x, prompter), num_proc = 8)
    return dataset
# %%
def dia_gen_process(example, prompter: Prompter):
    example['name'] = example['dataset'] + '-generation'

    dialog = dialog_process(
        history   = example['history'],
        utterance = example['utterance'],
        ai_last   = True
    )

    example['prompt'] = prompter.dia_gen_generate(
        dialog    = dialog,
        personas  = example['personas'],
        emotion   = example['emotion'],
        situation = example['situation'],
        act       = example['act']
    )

    return example

def dia_gen_sample_dataset(dataset, sample_nums, seed, prompter, do_sample=True):
    dataset = dataset.filter(lambda x: len(x['history']) % 2 == 1, num_proc=8)
    if do_sample:
        dataset = dataset.shuffle(seed)
        dataset = dataset.select(
            range( min(sample_nums, len(dataset)) )
        )
    dataset = dataset.map(lambda x: dia_gen_process(x, prompter), num_proc=8)
    return dataset
# %%
seed = 42
task_type_list = list(config.keys())

# make train data
datasets = []
for task_type in task_type_list:

    file_lists = list(config[task_type].keys())
    
    # prompter
    prompter = Prompter(template_name=task_type)

    for file in file_lists:
        data_path = os.path.join(task_type, file, 'train.jsonl')
        temp_set = load_dataset(
            'json',
            data_files = data_path,
            split      = 'train'
        )

        sample_number = config[task_type][file]
        
        if task_type == 'txt_clf':
            new_set = txt_clf_sample_dataset(
                temp_set, sample_number, seed=seed, prompter=prompter
            )

        elif task_type == 'dia_clf': 
            new_set = dia_clf_sample_dataset(
                temp_set, sample_number, seed=seed, prompter=prompter
            )

        elif task_type == 'er_abs':
            new_set = er_abs_sample_dataset(
                temp_set, sample_number, seed=seed, prompter=prompter
            )

        elif task_type == 'dia_inf':
            new_set = dia_inf_sample_dataset(
                temp_set, sample_number, seed=seed, prompter=prompter
            )

        elif task_type == 'dia_gen':
            new_set = dia_gen_sample_dataset(
                temp_set, sample_number, seed=seed, prompter=prompter
            )


        datasets.append(new_set)

dataset = concatenate_datasets(datasets)
stable_list = ['name', 'prompt']
remove_list = list(
    set(dataset.column_names) - set(stable_list)
)
dataset = dataset.remove_columns(
    remove_list
)
save_path = os.path.join('sample_train.jsonl')
dataset.to_json(save_path)

print(dataset)

print("done!")
# %%
