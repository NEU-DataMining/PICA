# %%
from transformers import set_seed, pipeline, LlamaForCausalLM, LlamaTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import os
from peft import PeftModel, prepare_model_for_int8_training
import torch
import pandas as pd
from datasets import load_dataset
import argparse
import datasets

parser = argparse.ArgumentParser(description='my infer scripts')


parser.add_argument('--model_path', type=str, default='/datas/huggingface/llama/llama-7b',
                    help='Path of the model')
parser.add_argument('--lora_path', type=str, default=None,
                    help='Path of the lora model')
parser.add_argument('--save_path', type=str, default=None,
                    help='Path of the save path')
parser.add_argument('--filename', type=str, default='er_abs-test.jsonl',
                    help='Name of the input file')
parser.add_argument('--do_lora', action='store_true', default=True,
                    help='Whether to use lora model')
parser.add_argument('--batch_size', type=int, default=6,
                    help='Batch size for inference')
parser.add_argument('--temperature', type=float, default=0.7,
                    help='Temperature for llama generation kwargs')
parser.add_argument('--sample', action='store_true', default=False,
                    help='Whether to do sample')

def load_model(path, device_map, do_lora = True, lora_path = None):
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path,
        load_in_8bit = True,
        device_map = device_map,
        torch_dtype = torch.float16,
    )
    model = prepare_model_for_int8_training(model)
    tokenizer = LlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=path,
        add_eos_token=True,
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'
    if do_lora:
        model = PeftModel.from_pretrained(
            model = model,
            model_id = lora_path,
            is_trainable = False,
            torch_dtype=torch.float16,
        )
    model.zero_grad()
    return model, tokenizer

args = parser.parse_args()
# base params
model_path  = args.model_path
lora_path   = args.lora_path
save_path   = args.save_path
filename    = args.filename
do_lora     = args.do_lora
sample      = args.sample
batch_size  = args.batch_size
task        = 'text-generation'
temperature = args.temperature

if lora_path is None:
    do_lora = False

device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

set_seed(42)

# generation params
generation_kwargs = {
    "num_beams"           : 1,
    "num_beam_groups"     : 1,
    "temperature"         : temperature,
    "top_p"               : 0.1,
    "top_k"               : 40,
    "max_new_tokens"      : 50,
    "num_return_sequences": 1,
    "return_full_text"    : False,
    "stop_sequence"       : ["\n"] if filename.startswith('dia_gen') else None,
    # "repetition_penalty"  : 1.3,
    # "early_stopping"      : True,
    # "diversity_penalty"   : 0.3,
    "prefix"              : "\n[ai]: " if filename.startswith('dia_gen') else None,
}

model, tokenizer = load_model(
    path       = model_path,
    device_map = device_map,
    do_lora    = do_lora,
    lora_path  = lora_path
)

model.config.pad_token_id = tokenizer.pad_token_id = 0
model.config.bos_token_id = 1
model.config.eos_token_id = 2

text_generator = pipeline(
    task         = task,
    model        = model,
    tokenizer    = tokenizer,
    pad_token_id = tokenizer.pad_token_id,
    device_map   = device_map,
)
# %%
data_files = os.path.join('datasets', 'testset', filename)

dataset = load_dataset(
    'json',
    data_files = data_files,
    split      = 'train'
)

if sample: 
    dataset = dataset.shuffle().select(range(100))

def gen_process(example):
    example['prompt'] = example['prompt']
    return example

dataset = dataset.map(gen_process, num_proc=8)

print(dataset)

results = []
predicts = []
for responses in tqdm(text_generator(
    KeyDataset(dataset, 'prompt'),
    batch_size = batch_size,
    **generation_kwargs,
), total=len(dataset)):
    for response in responses:
        results.append(response['generated_text'])
        predicts.append(response)

# save result
## post process & get predict
labels = list(dataset['output'])
names = list(dataset['name'])
prompts = list(dataset['prompt'])

df = pd.DataFrame({
    'name'   : names,
    'predict': predicts,
    'result' : results,
    'label'  : labels,
    'prompt' : prompts
})

df = datasets.Dataset.from_pandas(df)

save_path = os.path.join('evaluate', save_path)

if not os.path.exists(path=save_path):
    os.makedirs(save_path)


if sample:
    save_path = os.path.join(
        save_path, '{}-sample.json'.format(
            filename.split('-')[0], 
        )
    )
else:
    save_path = os.path.join(
        save_path, '{}.json'.format(
            filename.split('-')[0], 
        )
    )
    
df.to_json(save_path)