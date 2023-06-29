# %%
import os
import time

import pandas as pd
from tqdm.auto import tqdm

from datasets import load_dataset
from evaluate import load

from utils import ngrams

bert_score_metric = load('bertscore')

meteor_metric = load('meteor')

rouge_metric = load('rouge')

bleu_metric = load('bleu')

def coverage_area(A, B):
    if B in A:
        return len(B) / len(A)
    else:
        return 0
    
bert_score_metric, meteor_metric
# %%
root_path = 'llama-7b-emo-llm-12k'
data_path = 'er_abs.json'

dataset = load_dataset(
    'json', 
    data_files = os.path.join(root_path, data_path),
    split      = 'train'
)
dataset
# %%
from utils import ngrams
def er_abs_evaluate(result, label, name):
    bleu      = bleu_metric.compute(predictions=result, references=label)
    meteor    = meteor_metric.compute(predictions=result, references=label)
    bertscore = bert_score_metric.compute(predictions=result, references=label, lang='en')
    rouge     = rouge_metric.compute(predictions=result, references=label)
    cov = 0
    for r, l in zip(result, label):
        cov += coverage_area(r, l)
    return {
        'name'      : name,
        'bert-score': sum(bertscore['f1']) / len(bertscore['f1']),
        'meteor'    : meteor['meteor'],
        'rouge-l'   : rouge['rougeL'],
        'bleu'      : bleu['bleu'],
        'coverage-area': cov / len(result)
    }

def dia_gen_evaluate(result, label, name):
    bleu      = bleu_metric.compute(predictions=result, references=label)
    meteor    = meteor_metric.compute(predictions=result, references=label)
    bertscore = bert_score_metric.compute(predictions=result, references=label, lang='en')
    rouge     = rouge_metric.compute(predictions=result, references=label)
    dist_1    = distinct_n_corpus_level(result, n=1)
    dist_2    = distinct_n_corpus_level(result, n=2)
    dist_1_s  = distinct_n_sentence_level(' '.join(result), n=1)
    dist_2_s  = distinct_n_sentence_level(' '.join(result), n=2)

    length = 0
    for r in result:
        length += len(r.split())

    return {
        'name'        : name,
        'dist-1'      : dist_1,
        'dist-2'      : dist_2,
        'dist-1(lv-s)': dist_1_s,
        'dist-2(lv-2)': dist_2_s,
        'bert-score'  : sum(bertscore['f1']) / len(bertscore['f1']),
        'meteor'      : meteor['meteor'],
        'rouge-l'     : rouge['rougeL'],
        'bleu'        : bleu['bleu'],
        'ave length'  : length / len(result),
    }

def postprocess(dialogs):
    result = []
    prefix = "\n[ai]: "
    prefixt2 = "package com.example.android.sunshine.app.data;"
    for dialog in dialogs:
        if len(dialog) > 0:
            dialog = dialog.replace(prefix, "").split("\n")[0]
            dialog = dialog.replace(prefix, "")
            result.append(dialog)
        else:
            result.append(dialog)
    
    return result

def distinct_n_sentence_level(sentence, n):
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)

def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)

names = list(set(dataset['name']))

metrics = []

for name in tqdm(names, total=len(names)):
    temp_set = dataset.filter(lambda x: x['name'] == name, num_proc=8)
    
    label = temp_set['label']
    result = temp_set['result']
    if data_path.startswith('er_abs') or data_path.startswith('dia_inf'):
        r = er_abs_evaluate(result=result, label = label, name = name)
    elif data_path.startswith('dia_gen'):
        result = postprocess(result)
        r = dia_gen_evaluate(result=result, label=label, name = name)

    metrics.append(r)
# %% 
df = pd.DataFrame(metrics)

save_path = os.path.join(root_path, f'{data_path.split(".")[0]}.xlsx')

df.to_excel(save_path, index=False)

# %%

