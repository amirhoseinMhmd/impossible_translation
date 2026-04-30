import json
import argparse
from pathlib import Path
import sys

import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from tqdm import tqdm

# Add parent directory to path to allow imports from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.utils import (
    load_sentences_from_file,
    save_dataset,
    load_configs,
    get_device,
    apply_training_seed,
    split_training_data,
)
from utils.reverse import partial_reverse_batch
from utils.hop import wordhop_batch
from utils.shuffle import local_shuffle_batch, local_shuffle_batch_with_window, full_shuffle_batch

functions = {
    "partialReverse": partial_reverse_batch,
    "localShuffle": local_shuffle_batch,
    "localShuffle3": lambda texts: local_shuffle_batch_with_window(texts, window_size=3),
    "localShuffle5": lambda texts: local_shuffle_batch_with_window(texts, window_size=5),
    "fullShuffle": lambda texts: full_shuffle_batch(texts, seed=57),
    "wordHop": wordhop_batch
}

DEVICE = get_device()


def generate_training_data(input_file, type_of_perturbation):
    print("Generating training data...")
    sentences = load_sentences_from_file(input_file)
    training_data = functions[type_of_perturbation](sentences)
    return training_data


def process_single_chunk(input_text, tokenizer, model, max_position_embeddings):
    prompt = f"Fix this text: {input_text}\nCorrected:"

    prompt_encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = prompt_encoding['input_ids'].to(DEVICE)
    attention_mask = prompt_encoding['attention_mask'].to(DEVICE)

    input_tokens = tokenizer.encode(input_text)
    input_length = prompt_encoding['input_ids'].shape[1]

    max_new_tokens = min(len(input_tokens) + 5, max_position_embeddings - input_length)
    min_new_tokens = max(1, len(input_tokens) - 5)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Corrected:" in generated:
        corrected = generated.split("Corrected:")[1].strip()
    else:
        corrected = generated

    return corrected


def split_into_chunks(text, tokenizer, max_chunk_size=800, overlap=100):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    if len(tokens) <= max_chunk_size:
        return [{'text': text, 'start': 0, 'end': len(tokens)}]

    start = 0
    while start < len(tokens):
        end = min(start + max_chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append({'text': chunk_text, 'start': start, 'end': end})

        if end >= len(tokens):
            break
        start = end - overlap

    return chunks


def merge_chunks(chunks):
    if len(chunks) <= 1:
        return chunks[0] if chunks else ""

    result = chunks[0]
    for i in range(1, len(chunks)):
        result = result.rstrip() + " " + chunks[i].lstrip()

    return result


def process_long_text(input_corrupted, tokenizer, model, max_position_embeddings):
    input_tokens = tokenizer.encode(input_corrupted)
    prompt_template = "Fix this text: {}\nCorrected:"

    prompt_overhead = len(tokenizer.encode(prompt_template.format("")))
    max_input_size = max_position_embeddings - prompt_overhead - 50

    if len(input_tokens) <= max_input_size:
        return process_single_chunk(input_corrupted, tokenizer, model, max_position_embeddings)

    chunks = split_into_chunks(input_corrupted, tokenizer, max_chunk_size=max_input_size, overlap=100)
    corrected_chunks = []

    for chunk in chunks:
        corrected = process_single_chunk(chunk['text'], tokenizer, model, max_position_embeddings)
        corrected_chunks.append(corrected)

    return merge_chunks(corrected_chunks)


def generate_full_samples(model, tokenizer, sample_examples):
    max_position_embeddings = model.config.max_position_embeddings
    full_samples = []

    for input_corrupted, actual in tqdm(sample_examples, desc="Saving full samples", leave=False):
        if not input_corrupted:
            continue

        try:
            prediction = process_long_text(
                input_corrupted,
                tokenizer,
                model,
                max_position_embeddings,
            )
        except Exception as exc:
            print(f"Error generating sample: {exc}")
            prediction = ""

        full_samples.append({
            "input": input_corrupted,
            "prediction": prediction,
            "actual": actual,
        })

    return full_samples


class FullSamplesTrainer(Trainer):
    def __init__(self, *args, sample_examples=None, sample_output_dir=None, sample_output_prefix=None, export_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_examples = sample_examples or []
        self.sample_output_dir = Path(sample_output_dir or self.args.output_dir)
        self.sample_output_prefix = sample_output_prefix or "full_samples"
        self.export_tokenizer = export_tokenizer
        self.sample_output_dir.mkdir(parents=True, exist_ok=True)

    def save_full_samples(self, checkpoint_name):
        if not self.sample_examples or self.export_tokenizer is None:
            return

        was_training = self.model.training
        self.model.eval()
        full_samples = generate_full_samples(self.model, self.export_tokenizer, self.sample_examples)
        output_path = self.sample_output_dir / f"{self.sample_output_prefix}_{checkpoint_name}.json"
        save_dataset(full_samples, output_path)

        if was_training:
            self.model.train()

    def _save_checkpoint(self, model, trial):
        checkpoint_name = f"checkpoint-{self.state.global_step}"
        self.save_full_samples(checkpoint_name)


def prepare_dataset(training_data, tokenizer, train_split=0.9, max_length=128, split_seed=None):
    train_data, eval_data = split_training_data(
        training_data,
        train_split=train_split,
        split_seed=split_seed,
    )

    def process_data(data):
        """Process data with masked labels."""
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for corrupted, correct in tqdm(data):
            # INSTRUCTION FORMAT - tells GPT-2 what to do
            full_text = f"Fix this text: {corrupted}\nCorrected: {correct}<|endoftext|>"

            # Tokenize full text
            encoded = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors=None
            )

            # Find where "Corrected:" starts (where we want to start computing loss)
            prompt_text = f"Fix this text: {corrupted}\nCorrected:"
            prompt_encoded = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
                return_tensors=None
            )
            prompt_length = len(prompt_encoded['input_ids'])

            # Create labels with masking
            # -100 = ignored by loss function (prompt part)
            # actual token IDs = used for loss computation (output part)
            corrected_token_ids = tokenizer.encode("Corrected:", add_special_tokens=False)
            position = None
            for i in range(len(encoded['input_ids']) - len(corrected_token_ids)):
                if encoded['input_ids'][i:i + len(corrected_token_ids)] == corrected_token_ids:
                    position = i + len(corrected_token_ids)
                    break

            if position is None:
                position = prompt_length

            # Create labels
            labels = [-100] * position + encoded['input_ids'][position:]
            labels = labels[:max_length]
            if len(labels) < max_length:
                labels = labels + [-100] * (max_length - len(labels))

            input_ids_list.append(encoded['input_ids'])
            attention_mask_list.append(encoded['attention_mask'])
            labels_list.append(labels)

        return {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'labels': labels_list
        }

    print("Processing training data with masked labels...")
    train_processed = process_data(train_data)
    eval_processed = process_data(eval_data)

    # Create datasets
    train_dataset = Dataset.from_dict(train_processed)
    eval_dataset = Dataset.from_dict(eval_processed)

    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return train_dataset, eval_dataset, eval_data


def train_model(
        train_dataset,
        eval_dataset,
        eval_examples,
        config,
        model_name,
        dataset_name,
        type_of_perturbation,
        output_dir='./gpt2-reversal',
):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model = model.to(DEVICE)

    training_config = config.get('training_arguments', {}).copy()
    if training_config.get('load_best_model_at_end'):
        print("Disabling load_best_model_at_end because model checkpoints are not being saved.")
        training_config['load_best_model_at_end'] = False

    training_args = TrainingArguments(**training_config)

    trainer = FullSamplesTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        export_tokenizer=tokenizer,
        sample_examples=eval_examples,
        sample_output_dir=output_dir,
        sample_output_prefix=f"full_samples_{dataset_name}_{type_of_perturbation}",
    )

    print("Starting training...")
    trainer.train()
    trainer.save_full_samples("final")
    print(f"Saved full samples to {output_dir}")

    return model, tokenizer


def main(config, input_file, model_name, type_of_perturbation):
    apply_training_seed(config)
    max_length = config.get('data_arguments', {}).get('max_length', 128)
    OUTPUT_DIR = config.get('training_arguments', {}).get('output_dir', None)
    if not OUTPUT_DIR:
        raise ValueError("Output directory must be specified in training_arguments.output_dir")
    training_data_path = f"./training_data_{input_file.split('/')[-1].split('.')[0]}_{type_of_perturbation}.json"

    print(f"Reading sentences from {input_file}...")
    training_data = None
    if not Path(training_data_path).exists():
        print(f"{Path(training_data_path).resolve()} not found.\n Generating training data from {input_file}...")
        training_data = generate_training_data(
            input_file=input_file,
            type_of_perturbation=type_of_perturbation)
        save_dataset(training_data, training_data_path)
    else:
        print(f"Loading training data from {Path(training_data_path).resolve()}...")
        with open(training_data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)

    print("\nPreparing datasets...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset, eval_examples = prepare_dataset(
        training_data,
        tokenizer,
        max_length=max_length,
        split_seed=config.get('training_arguments', {}).get('data_seed'),
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    train_model(
        train_dataset,
        eval_dataset,
        eval_examples,
        config,
        model_name=model_name,
        dataset_name=Path(input_file).stem,
        type_of_perturbation=type_of_perturbation,
        output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, required=True,
                        help="Path to input sentences file")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument('-t', '--type', type=str, required=True,
                        help="Type of perturbation (wordHop, partialReverse, localShuffle3, localShuffle5, fullShuffle etc.)")

    args = parser.parse_args()
    config = load_configs(args.config)
    if args.type == 'wordHop':
        model = 'mission-impossible-lms/word-hop-gpt2'
    elif args.type == 'partialReverse':
        model = 'mission-impossible-lms/partial-reverse-gpt2'
    elif args.type == 'localShuffle3':
        model = 'mission-impossible-lms/local-shuffle-w3-gpt2'
    elif args.type == 'localShuffle5':
        model = 'mission-impossible-lms/local-shuffle-w5-gpt2'
    elif args.type == 'fullShuffle':
        model = 'mission-impossible-lms/deterministic-shuffle-s57-gpt2'
    else:
        raise ValueError("Invalid perturbation type. Choose from 'wordHop', 'partialReverse', 'localShuffle', 'localShuffle3', 'localShuffle5', or 'fullShuffle'.")
    main(config=config, input_file=args.path, model_name=model, type_of_perturbation=args.type)
