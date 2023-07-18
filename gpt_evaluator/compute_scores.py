import argparse
import concurrent.futures
import json
import os
import time

from datasets import load_dataset
from evaluator import Evaluator
from tqdm.auto import tqdm
from utils import get_azure_response, parse_output


def run(content, n):
    retry_numbers = 0
    while True:
        retry_numbers += 1
        response = get_azure_response(
            url      = url,
            apikey   = apikey,
            content  = content,
            n        = n,
        )

        all_scores = [parse_output(r) for r in response]

        count = 0
        scores = 0
        for score in all_scores:
            if score > 0 and score <= 5:
                count += 1
                scores += score

        if count >= 2/3 * n or (retry_numbers >= 5 and count > 0):
            break

    return scores / count

def get_dataset(path):
    dataset = load_dataset(
        'json',
        data_files = path,
        split      = 'train'
    )

    histories = dataset['history']
    prompts = dataset['prompt']
    responses = dataset['result']

    return prompts, responses, histories, dataset

parser = argparse.ArgumentParser(description="evaluate by chatgpt")

parser.add_argument('--type', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--output_path', type=str, default='scores.json')
parser.add_argument('--url', type=str)
parser.add_argument('--apikey', type=str)

args = parser.parse_args()


if __name__ == '__main__':
    apikey = args.apikey
    url = args.url

    prompts, responses, histories, dataset = get_dataset(args.data_path)

    evaluator = Evaluator(type=args.type)
    queries = evaluator.make_queries(
        prompts   = prompts,
        responses = responses,
        histories = histories
    )
    print(queries[8])
    # time.sleep(100)

    fp = open(os.path.join('result', args.output_path), 'a', encoding='utf-8')
    total_scores = 0
    total_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                run,
                content = query,
                n       = 20
            ) for query in queries
        ]

        with tqdm(total=len(futures)) as pbar:
            for future, data in zip(concurrent.futures.as_completed(futures), dataset):
                try:
                    data['scores'] = future.result()
                    total_scores += data['scores']
                    total_count += 1
                    fp.write(
                        json.dumps(data, ensure_ascii=False) + '\n'
                    )
                    pbar.update(1)
                    pbar.set_postfix({f'{args.type.upper()} Average Score': total_scores/total_count})
                except Exception as e:
                    print(e)
                    pbar.update(1)

    print(f'{args.data_path}')
    print(f'{args.type} score: {total_scores / total_count}')

    with open('./result/result.json', 'a') as w:
        w.write(
            json.dumps({
                'data': args.data_path,
                'type': args.type,
                'score': total_scores / total_count
            }, ensure_ascii=False) + '\n'
        )



    


