import argparse
import json
import os

from datasets import Dataset, DatasetDict
import utils

def preprocess(tokenizer):
    def tokenize(examples):
        inputs = tokenizer(
            examples['perturbed_text'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors=None
        )
        labels = tokenizer(
            examples['original_text'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors=None
        )['input_ids']

        inputs['labels'] = labels
        return inputs

    return tokenize


def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_dataset = Dataset.from_list(data["train"])
    valid_dataset = Dataset.from_list(data["validate"])

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": valid_dataset
    })

    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, required=True,
                        help="Path to file")
    parser.add_argument('-t', '--type', type=str, required=True,
                        help="Type of perturbation")
    args = parser.parse_args()

    data = load_dataset(args.path)
    perturb_type = args.type

    tokenizer = None

    if perturb_type == 'hop':
        tokenizer = utils.gpt2_hop_tokenizer
    elif perturb_type == 'reverse':
        tokenizer = utils.gpt2_rev_tokenizer
    elif perturb_type == 'shuffle':
        tokenizer = utils.gpt2_original_tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenize = preprocess(tokenizer)

    dataset = data.map(
        tokenize,
        batched=True,
        batch_size=2000,
        remove_columns=['perturbed_text', 'original_text'],
        num_proc=8,
        load_from_cache_file=True,  # Use cache if available
        desc="Tokenizing dataset"
    )

    dataset.save_to_disk(f"{os.path.dirname(args.path)}/tokenized_dataset")