# %%
from datasets import load_dataset
from prompters import Prompter

def preprocess(example):
    """
    input: instruct + instances['input']
    """
    instruction = example['instruction']
    input = example['instances'][0]['input']
    output = example['instances'][0]['output']
    example['input'] = f"{instruction} {input}"
    example['output'] = f"{output}"
    return example

def generate_prompt(example):
    instruction = example["instruction"]
    input = example['instances'][0]['input']
    output = example['instances'][0]['output']
    if 'task_type' in example.keys():
        prompter = Prompter(template_name=example['task_type'])
    else:
        prompter = Prompter()
    
    prompt = prompter.generate_prompt(
        instruction=instruction,
        input=input,
        label=output,
    )

    example['input'] = prompt
    example['output'] = output
    return example

def get_dataset(data_path, key=None, value=None):
    dataset = load_dataset(
        'json',
        data_files={
            'train': data_path
        },
        split = 'train'
    )
    dataset = dataset.filter(lambda x: x['name'] != 'emowoz-classification')
    if key is not None:
        dataset = dataset.filter(lambda x: x[key] == value)

    dataset = dataset.map(lambda x: generate_prompt(x), num_proc=8)
    # dataset = dataset.map(lambda x: preprocess(x), num_proc=8)

    stable_list = ['name', 'input', 'output']
    remove_list = list(set(dataset.column_names) - set(stable_list))
    dataset = dataset.remove_columns(remove_list)

    return dataset

if __name__ == '__main__':
    dataset = get_dataset(
        data_path = 'dataset/train.jsonl'
    )
    print(dataset)
    temp_set = dataset.shuffle().select(range(20))
    temp_set.to_json(path_or_buf='temp_watch.json', indent=2)
# %%

# dataset = get_dataset('dataset/train.jsonl')
# %%
# import tiktoken
# from tqdm import tqdm
# enc = tiktoken.encoding_for_model("gpt-4")

# d = {}
# count = 0
# for data in tqdm(dataset):
#     data = data['input']
#     encode = enc.encode(data)
#     if len(encode) > 512:
#         count += 1
# count
# # %%
# dataset[0]
# dataset = load_dataset(
#     'json',
#     data_files='dataset/train.jsonl',
#     split='train'
# )

# import re
# dataset = dataset.filter(
#     lambda x: x['task_type'] == 'dig_clf'
# )

# dataset = dataset.filter(
#     lambda x: x['name'] != 'emowoz-classification'
# )
# dataset
# # %%
# for data in dataset:
#     input = data['instances'][0]['input']
#     match = re.search(r"History: (.*) Utterance: (.*)", input)
#     try:
#         history = match.group(1)
#         utterance = match.group(2)
#     except AttributeError as e:
#         print(e)
#         print(input)

# %%
