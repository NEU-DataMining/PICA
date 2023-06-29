# %%
from datasets import load_dataset, DatasetDict, Dataset
from typing import List, Dict
import os

path = './dia_gen'
dir_names = next(os.walk(path))[1]

print(dir_names)
# %%
txt_clf = ['amazon_polarity', 'cams', 'imdb', 'financial', 'emocontext', 'yelp', 'hotel', 'emotion-stimulus', 'ntcir-13_eca']
dig_clf = ['edos', 'emorynlp', 'emotionlines', 'emowoz', 'empatheticdialogues', 'goemotion', 'iemocap', 'mastodon', 'meld', 'memor']
er_abs = ['DailyDialog', 'emotion-stimulus', 'goodnewseveryone', 'ntcir-13_eca']
dia_inf = ['dailydialog', 'empatheticdialogues', 'mastodon-act', 'esconv', 'personaminedit']
dia_gen = ['DailyDialog', 'edos', 'emorynlp', 'emotionlines', 'emowoz', 'empatheticdialogues', 'ESConv', 'mastodon', 'meld', 'memor', 'personaminedit']
# %%
## txt_clf: {dataset, input, output}
## dig_clf: {dataset, history, utterance, output}
## er_abs : {dataset, input, cause, emotion}  
## dia_inf: {dataset, history, utterance, output}
## dia_gen: {dataset, history, utterance, personas, situation, emotion, act}

file_path = os.path.join(path, dia_gen[10])

data_files = {
    'train': os.path.join(file_path, 'train.jsonl'),
    'test': os.path.join(file_path, 'test.jsonl')
}

train_set = load_dataset(
    'json', data_files=data_files['train'], split='train'
)

test_set = load_dataset(
    'json', data_files=data_files['test'], split='train'
)

dataset = DatasetDict({
    'train': train_set,
    'test': test_set
}) 
print(file_path)
dataset
# %%
def pre(example):
    example['personas'] = "none"
    return example
dataset = dataset.map(pre, num_proc=8)

dataset
# %%
def pro(example):
    example['output'] = example['output'][0]
    return example

def inplace_label(example):
    example['emo_label'] = example['emo_candidate'][example['emo_label']]
    return example

def er_process(example):
    outs = []
    for output in example['output']:
        cause = output[0]
        emotion = output[-1]

        out = {
            'emotion': emotion,
            'cause': cause
        }

        outs.append(out)

    example['output'] = outs
    input = '\n'.join([dialog[0] for dialog in example['input']])
    example['input'] = input
    return example

def gen_process(example):
    if len(example['personas']) == 0:
        example['personas'] = 'none'
    else:
        example['personas'] = example['personas'][0]

    if 'emotion' not in example.keys():
        example['emotion'] = "none"
    if 'act' not in example.keys():
        example['act'] = "none"
    if 'personas' not in example.keys():
        example['personas'] = "none"
    if 'situation' not in example.keys():
        example['situation'] = "none"

    if example['emotion'] == "":
        example['emotion'] = 'none'
    if example['act'] == "":
        example['act'] = 'none'
    if example['personas'] == "":
        example['personas'] = 'none'
    if example['situation'] == "":
        example['situation'] = 'none'

    return example

def remove_column(dataset: Dataset, rename_map: List, stable_list: List):
    dataset = dataset.filter(lambda x: len(x['personas']) > 0)
    dataset = dataset.rename_columns(rename_map)
    dataset = dataset.map(gen_process, num_proc=8)
    remove_list = list(set(dataset.column_names) - set(stable_list))
    dataset = dataset.remove_columns(remove_list)


    if 'input' in dataset.column_names:
        dataset = dataset.filter(lambda x: len(x['input']) > 0)
    else:
        dataset = dataset.filter(lambda x: len(x['history']) > 0)
        dataset = dataset.filter(lambda x : len(x['utterance']) > 0)
    
    return dataset

rename_map = {
    # 'emo_label': 'emotion',
    # 'strategy': 'act',
    'input': 'history',
    'output': 'utterance'
    # 'utterance': 'input'
}

stable_list = ['dataset', 'act', 'emotion', 'personas', 'situation', 'history', 'utterance']
train_set = remove_column(dataset['train'], rename_map=rename_map, stable_list=stable_list)
test_set = remove_column(dataset['test'], rename_map=rename_map, stable_list=stable_list)
train_set, test_set
# %%
train_set[0]
# %%
import json

with open('er_abs/goodnewseveryone/train.json') as f:
    with open('er_abs/goodnewseveryone/train.jsonl', 'w') as fout:  # 打开新文件
        for line in f:
            data = json.loads(line)
            output = data.pop('output')  # 获取output并从原数据中删除
            for item in output:
                new_data = dict(data)
                new_data['input'] = new_data['input'][0]
                new_data['cause'] = item['cause']
                new_data['emotion'] = item['emotion']
                # 处理新数据
                new_line = json.dumps(new_data) + '\n'  # 转换成json字符串并添加换行符
                fout.write(new_line)  # 将新数据写入新文件
# %%
train_set.filter(lambda x: len(x['output']) > 1)
# %%
# coarse <-> fine
edos_coarse_map = {
    'love': ['anticipating', 'faithful', 'hopeful', 'caring', 'trusting', 'nostalgic', 'wishing', 'consoling', 'encouraging', 'sentimental'],
    'joy': ['prepared', 'proud', 'excited', 'joyful', 'content', 'confident', 'agreeing', 'acknowledging', 'neutral', 'questioning', 'suggesting'],
    'surprise': ['grateful', 'surprised', 'impressed', 'devastated'],
    'anger': ['disappointed', 'disgusted', 'furious', 'angry', 'annoyed', 'jealous'],
    'sadness': ['embarrassed', 'ashamed', 'sad', 'lonely', 'guilty', 'sympathizing'],
    'fear' : ['anxious', 'terrified', 'afraid', 'apprehensive'],
}

ed_coarse_map = {
    'love': ['anticipating', 'faithful', 'hopeful', 'caring', 'trusting', 'nostalgic', 'wishing', 'consoling', 'encouraging', 'sentimental'],
    'joy': ['prepared', 'proud', 'excited', 'joyful', 'content', 'confident', 'agreeing', 'acknowledging', 'neutral', 'questioning', 'suggesting'],
    'surprise': ['grateful', 'surprised', 'impressed', 'devastated'],
    'anger': ['disappointed', 'disgusted', 'furious', 'angry', 'annoyed', 'jealous'],
    'sadness': ['embarrassed', 'ashamed', 'sad', 'lonely', 'guilty', 'sympathizing'],
    'fear' : ['anxious', 'terrified', 'afraid', 'apprehensive'],
}

goemotion_coarse_map = {
    'love': ['admiration', 'approval', 'caring', 'desire', 'love'],
    'joy': ['amusement', 'curiosity', 'excitement', 'joy', 'optimism', 'pride', 'realization', 'relief', 'neutral'],
    'surprise': ['surprise', 'gratitude'],
    'anger': ['anger', 'annoyance', 'disgust', 'disappointment', 'disapproval'],
    'sadness': ['embarrassment', 'grief', 'confusion', 'sadness'],
    'fear': ['fear', 'nervousness', 'remorse']
}

memor_coarse_map = {
    'neutral': ['neutral', 'serenity', 'anticipation', 'surprise'],
    'sadness': ['sadness', 'boredom', 'fear'],
    'positive': ['joy', 'interest', 'trust'],
    'anger': ['annoyance', 'distraction', 'anger', 'disgust'],
}

def get_coarse_key(label, map):
    for key, value in map.items():
        if label in value:
            return key
    raise KeyError(f'{label} error')

def coarse_process(example):
    example['output'] = get_coarse_key(example['output'], memor_coarse_map)

    return example

train_set_coarse = train_set.map(lambda x: coarse_process(x))
test_set_coarse = test_set.map(lambda x: coarse_process(x))

train_set_coarse, test_set_coarse
# %%
# %%
from collections import Counter

print(dict(Counter(train_set['output'])))
print(dict(Counter(test_set['output'])))
# print(dict(Counter(train_set_coarse['output'])))
# print(dict(Counter(test_set_coarse['output'])))
# %%
def save_dataset(train_set: Dataset, test_set: Dataset, path_map: Dict):
    train_set.to_json(path_map['train'])
    test_set.to_json(path_map['test'])

# data_files = {
#     'train': 'dailydialog/train.jsonl',
#     'test': 'dailydialog/test.jsonl'
# }

coarse_data_files = {
    'train': 'memor-coarse/train.jsonl',
    'test': 'memor-coarse/test.jsonl'
}
print(data_files)
save_dataset(train_set, test_set, data_files)
# save_dataset(train_set_coarse, test_set_coarse, coarse_data_files)
# %%
save_dataset(dataset['train'], dataset['test'], data_files)
# %%
# %%
'none'
# %%
from datasets import load_dataset

edos_coarse_map = {
    'love': ['anticipating', 'faithful', 'hopeful', 'caring', 'trusting', 'nostalgic', 'wishing', 'consoling', 'encouraging', 'sentimental'],
    'joy': ['prepared', 'proud', 'excited', 'joyful', 'content', 'confident', 'agreeing', 'acknowledging', 'neutral', 'questioning', 'suggesting'],
    'surprise': ['grateful', 'surprised', 'impressed', 'devastated'],
    'anger': ['disappointed', 'disgusted', 'furious', 'angry', 'annoyed', 'jealous'],
    'sadness': ['embarrassed', 'ashamed', 'sad', 'lonely', 'guilty', 'sympathizing'],
    'fear' : ['anxious', 'terrified', 'afraid', 'apprehensive'],
}

ed_coarse_map = {
    'love': ['anticipating', 'faithful', 'hopeful', 'caring', 'trusting', 'nostalgic', 'wishing', 'consoling', 'encouraging', 'sentimental'],
    'joy': ['prepared', 'proud', 'excited', 'joyful', 'content', 'confident', 'agreeing', 'acknowledging', 'neutral', 'questioning', 'suggesting'],
    'surprise': ['grateful', 'surprised', 'impressed', 'devastated'],
    'anger': ['disappointed', 'disgusted', 'furious', 'angry', 'annoyed', 'jealous'],
    'sadness': ['embarrassed', 'ashamed', 'sad', 'lonely', 'guilty', 'sympathizing'],
    'fear' : ['anxious', 'terrified', 'afraid', 'apprehensive'],
}

goemotion_coarse_map = {
    'love': ['admiration', 'approval', 'caring', 'desire', 'love'],
    'joy': ['amusement', 'curiosity', 'excitement', 'joy', 'optimism', 'pride', 'realization', 'relief', 'neutral'],
    'surprise': ['surprise', 'gratitude'],
    'anger': ['anger', 'annoyance', 'disgust', 'disappointment', 'disapproval'],
    'sadness': ['embarrassment', 'grief', 'confusion', 'sadness'],
    'fear': ['fear', 'nervousness', 'remorse']
}

memor_coarse_map = {
    'neutral': ['neutral', 'serenity', 'anticipation', 'surprise'],
    'sadness': ['sadness', 'boredom', 'fear'],
    'positive': ['joy', 'interest', 'trust'],
    'anger': ['annoyance', 'distraction', 'anger', 'disgust'],
}

def get_coarse_key(label, map):
    for key, value in map.items():
        if label in value:
            return key
    raise KeyError(f'{label} error')

data_files = {
    "train": "txt_clf/goemotion-fine/train.jsonl",
    "test": "txt_clf/goemotion-fine/test.jsonl"
}

dataset = load_dataset(
    'json',
    data_files=data_files,
)

dataset, dataset['train'][0]
# %%
def fine_process(example):
    fine_key = get_coarse_key(example['output'], goemotion_coarse_map)

    example['dataset'] = 'GoEmotion' + '-' + fine_key
    # example['dataset'] = 'memor-coarse'
    return example

dataset = dataset.map(fine_process, num_proc=16)
dataset,dataset['train'][0]
# %%
from datasets import Dataset
from typing import Dict
def save_dataset(train_set: Dataset, test_set: Dataset, path_map: Dict):
    train_set.to_json(path_map['train'])
    test_set.to_json(path_map['test'])

save_dataset(dataset['train'], dataset['test'], data_files)
# %%
from datasets import load_dataset
import tiktoken

enc = tiktoken.get_encoding('cl100k_base')
enc
# %%
dataset = load_dataset(
    'json',
    data_files='./testset/dia_clf-test.jsonl',
    split='train'
)
# %%
count = 0
d = {}
for data in dataset:
    coder = enc.encode(data['prompt'])
    if len(coder) > 1024:
        print(data['name'], len(coder))
        if data['name'] not in d.keys():
            d[data['name']] = 1
        else:
            d[data['name']] += 1
        count += 1
count, d
# %%
# %%
