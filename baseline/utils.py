from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import BatchEncoding, EvalPrediction, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy

import evaluate

@dataclass
class DataCollatorForSequenceClassification:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_if: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        if 'label' in features[0].keys():
            labels: List[str] = [
                feature.pop('label') for feature in features
            ]
        else:
            labels = None
        
        padded_features: BatchEncoding = self.tokenizer.pad(
            features,
            padding            = self.padding,
            max_length         = self.max_length,
            pad_to_multiple_of = self.pad_to_multiple_if,
            return_tensors     = 'pt'
        )

        batch: Dict[str, torch.Tensor] = {
            k: v for k, v in padded_features.items()
        }

        if labels is not None:
            batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits, label = eval_pred
    predictions = np.argmax(logits, axis = -1)

    acc_metric = evaluate.load('accuracy')
    f1_metric = evaluate.load('f1')
    accuracy = acc_metric.compute(predictions=predictions, references=label)

    macro_f1 = f1_metric.compute(
        references  = label,
        predictions = predictions,
        average     = 'macro'
    )
    micro_f1 = f1_metric.compute(
        references  = label,
        predictions = predictions,
        average     = 'macro'
    )
    weighted_f1 = f1_metric.compute(
        references  = label,
        predictions = predictions,
        average     = 'weighted'
    )

    return {
        'macro_f1'   : macro_f1['f1'],
        'micro_f1'   : micro_f1['f1'],
        'weighted_f1': weighted_f1['f1'],
        'accuracy'   : accuracy['accuracy'],
    }

def txt_preprocess(example, label2id: Dict, tokenizer: PreTrainedTokenizerBase):
    tokenizer_examples: BatchEncoding = tokenizer(
        example['input'],
        truncation = True,
    )
    example['label'] = label2id[example['output']]
    return tokenizer_examples 

def dia_preprocess(example, label2id: Dict, tokenizer: PreTrainedTokenizerBase):
    # example.keys() : dataset, history: List, utterance: str, output
    history = f'{tokenizer.sep_token}'.join(example['history'] + [example['utterance']])

    tokenizer_examples: BatchEncoding = tokenizer(
        history,
        truncation=True,
    )
    
    example['label'] = label2id[example['output']]

    return tokenizer_examples