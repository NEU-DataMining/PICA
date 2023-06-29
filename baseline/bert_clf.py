import argparse
import os

import torch
import transformers
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, set_seed)
from utils import *

from datasets import load_dataset


def train(
    # model/data params
    model_path: str,
    data_path: str,
    output_dir: str,
    ddp: bool = True,
    # training params
    batch_size: int = 256,
    micro_batch_size: int = 32,
    num_epochs: int = 20,
    learning_rate: float = 1e-5,
    fp16: bool = True,
    val_set_size: int = 2000,
    # wandb params
    seed: int = 42,
    wandb_project: str = "",
    wandb_run_name: str = "",
):  
    set_seed(seed=seed)

    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = 'auto'
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # load data
    dataset = load_dataset(
        'json',
        data_files={
            'train': data_path,
            'test' : data_path.replace("train", "test")
        }
    )

    ## get label map
    label_list = list(set(dataset['train']['output']))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i:label for i, label in enumerate(label_list)}

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path
    )
    ## process data -> (dataset, input/(history, dialog), output)

    if 'txt_clf' in data_path:
        dataset = dataset.map(
            lambda x: txt_preprocess(
                x, label2id=label2id, tokenizer=tokenizer
            ), 
            num_proc = 8,
        )
    elif 'dia_clf' in data_path:
        dataset = dataset.map(
            lambda x: dia_preprocess(
                x, label2id=label2id, tokenizer=tokenizer
            ), 
            num_proc = 8,
        )

    dataset, testset = dataset['train'], dataset['test']

    if val_set_size > 0:
        val_set_size = int(len(dataset) * val_set_size)

        dataset = dataset.train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
        )
        train_data = (
            dataset['train'].shuffle(seed=seed)
        )
        val_data = (
            dataset['test'].shuffle(seed=seed)
        )
    else:
        train_data = dataset.shuffle(seed=seed)
        val_data = None

    print(train_data, val_data, testset)

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_path,
        device_map = device_map,
        num_labels = len(label2id),
    )
    model.config.label2id = label2id
    model.config.id2label = id2label

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # Training
    trainer = transformers.Trainer(
        model = model,
        train_dataset = train_data,
        eval_dataset = val_data,
        args = transformers.TrainingArguments(
            per_device_train_batch_size = micro_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            warmup_ratio                = 0.1,
            # warmup_steps                = 200,
            num_train_epochs            = num_epochs,
            learning_rate               = learning_rate,
            fp16                        = fp16,
            logging_steps               = 200,
            evaluation_strategy         = "epoch" if val_set_size > 0 else "no",
            save_strategy               = "epoch",
            output_dir                  = output_dir,
            save_total_limit            = 3,
            load_best_model_at_end      = True if val_set_size > 0 else False,
            ddp_find_unused_parameters  = False if ddp else None,
            metric_for_best_model       = 'eval_macro_f1',
            greater_is_better           = True,
            label_names                 = ['labels'],
            optim                       = "adamw_torch",
            report_to                   = "wandb" if use_wandb else None,
            run_name                    = wandb_run_name if use_wandb else None,
        ),
        data_collator   = DataCollatorForSequenceClassification(tokenizer = tokenizer),
        compute_metrics = lambda x: compute_metrics(x),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    trainer.train()
    # add test
    print(testset)
    predictions = trainer.predict(test_dataset=testset)

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    trainer.save_metrics(metrics = predictions.metrics, split='test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some parameters.')

    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--data_path', type=str, help='Path to the data')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--ddp', action='store_true', help='Enable distributed data parallel')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--micro_batch_size', type=int, default=32, help='Micro batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--fp16', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--val_set_size', type=float, default=0.0, help='Size of the validation set')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb_project', type=str, default='', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='', help='Wandb run name')
    parser.add_argument('--local_rank', type=int, default=-1, help="node rank")

    args = parser.parse_args()

    train(
        model_path       = args.model_path,
        data_path        = args.data_path,
        output_dir       = args.output_dir,
        ddp              = args.ddp,
        batch_size       = args.batch_size,
        micro_batch_size = args.micro_batch_size,
        num_epochs       = args.num_epochs,
        learning_rate    = args.learning_rate,
        fp16             = args.fp16,
        val_set_size     = args.val_set_size,
        seed             = args.seed,
        wandb_project    = args.wandb_project,
        wandb_run_name   = args.wandb_run_name
    )
