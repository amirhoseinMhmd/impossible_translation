import argparse

import torch
import yaml
from datasets import load_dataset
from transformers import (GPT2LMHeadModel, GPT2Config,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)

import utils

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


def load_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_gpt_from_scratch(config, vocab_size):
    model_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=config.get('n_positions', 1024),  # Context length
        n_embd=config.get('n_embd', 768),  # Embedding dimension
        n_layer=config.get('n_layer', 12),  # Number of transformer layers
        n_head=config.get('n_head', 12),  # Number of attention heads
        n_inner=config.get('n_inner', 3072),  # Feed-forward dimension
        activation_function=config.get('activation_function', 'gelu_new'),
        resid_pdrop=config.get('resid_pdrop', 0.1),
        embd_pdrop=config.get('embd_pdrop', 0.1),
        attn_pdrop=config.get('attn_pdrop', 0.1),
        layer_norm_epsilon=config.get('layer_norm_epsilon', 1e-5),
        initializer_range=config.get('initializer_range', 0.02),
        summary_type=config.get('summary_type', 'cls_index'),
        summary_use_proj=config.get('summary_use_proj', True),
        summary_activation=config.get('summary_activation', None),
        summary_proj_to_labels=config.get('summary_proj_to_labels', True),
        summary_first_dropout=config.get('summary_first_dropout', 0.1),
        use_cache=config.get('use_cache', True),
        bos_token_id=config.get('bos_token_id', 50256),
        eos_token_id=config.get('eos_token_id', 50256),
    )

    print(f"Creating model from scratch with configuration:")
    print(f"  - Vocab size: {vocab_size}")
    print(f"  - Embedding dimension: {model_config.n_embd}")
    print(f"  - Number of layers: {model_config.n_layer}")
    print(f"  - Number of attention heads: {model_config.n_head}")
    print(f"  - Context length: {model_config.n_positions}")

    model = GPT2LMHeadModel(config=model_config)
    model.apply(model._init_weights)

    return model

def tokenize_function(examples):
    inputs = tokenizer(
        examples['perturbed_text'],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    inputs["labels"] = inputs["input_ids"][:]
    return inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, required=True,
                        help="Path to file")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument('-t', '--type', type=str, required=True,
                        help="Type of perturbation")

    args = parser.parse_args()
    configs = load_configs(args.config)
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

    vocab_size = len(tokenizer)

    model = create_gpt_from_scratch(configs.get('model_config', {}), vocab_size)
    model.to(device)

    train_dataset = load_dataset(
        "json",
        data_files=args.path,
        field="train",
        split="train"
    )

    eval_dataset = load_dataset(
        "json",
        data_files=args.path,
        field="validation",
        split="train"
    )

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True,remove_columns=train_dataset.column_names)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(**configs.get('training_arguments', {}))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset= tokenized_eval_dataset
    )

    trainer.train()
    model = model.merge_and_unload()
    model.save_pretrained('./fine_tuned_model')
