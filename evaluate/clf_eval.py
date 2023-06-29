# %%
import os
import time

import pandas as pd
from tqdm.auto import tqdm

from datasets import load_dataset
from evaluate import load

def make_label_map(label):
    label = set(label)
    label_value = range(len(label))

    label_map = {k:v for k, v in zip(label, label_value)}

    unknown_map = {'unknown': -1}
    label_map.update(unknown_map)

    return label_map

def get_result_idx(label_map, result):
    res = []
    for r in result:
        if r in label_map.keys():
            res.append(label_map[r])
        else:
            res.append(-1)
    return res


def compute_clf_metrics(label, result, name):
    label_map = make_label_map(label)
    label_idx = [label_map[l] for l in label]
    result_idx = get_result_idx(label_map, result)

    metric_f1 = load('f1')
    metric_acc = load('accuracy')

    macro_f1 = metric_f1.compute(
        references  = label_idx,
        predictions = result_idx,
        average     = 'macro'
    )
    micro_f1 = metric_f1.compute(
        references  = label_idx,
        predictions = result_idx,
        average     = 'micro'
    )
    weighted_f1 = metric_f1.compute(
        references  = label_idx,
        predictions = result_idx,
        average     = 'weighted'
    )
    accuracy = metric_acc.compute(references=label_idx, predictions = result_idx)
    error_label_list = []
    for r, i in zip(result, result_idx):
        if i == -1 and r not in error_label_list:
            error_label_list.append(r)

    return {
        'name'            : name,
        'testset size'    : len(label),
        'label number'    : len(label_map),
        'macro f1'        : macro_f1['f1'],
        'micro f1'        : micro_f1['f1'],
        'weighted f1'     : weighted_f1['f1'],
        'accuracy'        : accuracy['accuracy'],
        'error label'     : result_idx.count(-1),
        'error label list': error_label_list
    }

root_path = 'llama-7b-emo-llm-12k'

data_path = 'txt_clf.json'


dataset = load_dataset(
    'json',
    data_files = os.path.join(root_path, data_path),
    split      = 'train',
)

names = list(set(dataset['name']))

metrics = []

for name in tqdm(names, total=len(names)):
    temp_set = dataset.filter(lambda x: x['name'] == name, num_proc=8)
    label    = temp_set['label']
    result   = temp_set['result']
    metrics.append(
        compute_clf_metrics(label=label, result=result, name=name)
    )

# %%
df = pd.DataFrame(metrics)

save_path = os.path.join(root_path, f'{data_path.split(".")[0]}.xlsx')

df.to_excel(save_path, index=False)
# %%
