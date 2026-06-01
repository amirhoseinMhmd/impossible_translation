import argparse
import json
import os
from pathlib import Path

import numpy as np
import spacy
from tqdm import tqdm

GPU_ENABLED = False
NLP_MODEL_NAME = "en_core_web_sm"

try:
    GPU_ENABLED = spacy.prefer_gpu()
except Exception:
    GPU_ENABLED = False

try:
    nlp = spacy.load("en_core_web_trf")
    NLP_MODEL_NAME = "en_core_web_trf"
except OSError:
    nlp = spacy.load("en_core_web_sm")
    NLP_MODEL_NAME = "en_core_web_sm"

DEFAULT_BATCH_SIZE = int(os.environ.get("SPACY_PIPE_BATCH_SIZE", "64"))

SPECIAL_TOKENS = {"R", "S", "P"}


def avg_dependency_length(doc):
    lengths = [abs(token.i - token.head.i) for token in doc if token.dep_ != "ROOT"]
    return float(np.mean(lengths)) if lengths else 0.0


def get_dep_triples(doc):
    triples = set()
    for token in doc:
        if token.dep_ != "ROOT":
            triples.add((token.text.lower(), token.dep_, token.head.text.lower()))
    return triples


def dep_f1(doc_a, doc_b):
    a = get_dep_triples(doc_a)
    b = get_dep_triples(doc_b)

    if not a or not b:
        return 0.0

    matched = a & b
    precision = len(matched) / len(b) if b else 0.0
    recall = len(matched) / len(a) if a else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def build_token_mapping(actual_tokens, perturbed_tokens):
    clean_perturbed = [
        (i, tok) for i, tok in enumerate(perturbed_tokens)
        if tok not in SPECIAL_TOKENS
    ]
    used = set()
    mapping = {}

    for orig_idx, orig_tok in enumerate(actual_tokens):
        for pert_idx, pert_tok in clean_perturbed:
            if pert_tok == orig_tok and pert_idx not in used:
                mapping[orig_idx] = pert_idx
                used.add(pert_idx)
                break

    return mapping


def remapped_adl(original_doc, perturbed_text):
    actual_tokens = [t.text for t in original_doc]
    perturbed_tokens = perturbed_text.split()
    mapping = build_token_mapping(actual_tokens, perturbed_tokens)

    lengths = []
    for token in original_doc:
        if token.dep_ == "ROOT":
            continue
        if token.i in mapping and token.head.i in mapping:
            lengths.append(abs(mapping[token.i] - mapping[token.head.i]))

    return float(np.mean(lengths)) if lengths else 0.0


def load_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for i, item in enumerate(data):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Item {i} is not [perturbed, original]")
        perturbed, original = item
        pairs.append((perturbed, original))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    print(
        f"spaCy model: {NLP_MODEL_NAME} | "
        f"GPU enabled: {GPU_ENABLED} | "
        f"batch size: {args.batch_size}"
    )

    pairs = load_pairs(args.input)
    perturbed_texts = [p for p, o in pairs]
    original_texts = [o for p, o in pairs]

    # Batch-parse all texts in two passes through nlp.pipe()
    print("Parsing perturbed texts...")
    perturbed_docs = list(tqdm(
        nlp.pipe(perturbed_texts, batch_size=args.batch_size),
        total=len(perturbed_texts),
        desc="Perturbed",
    ))

    print("Parsing original texts...")
    original_docs = list(tqdm(
        nlp.pipe(original_texts, batch_size=args.batch_size),
        total=len(original_texts),
        desc="Original",
    ))

    # Compute metrics using pre-parsed docs
    perturbed_adls = []
    original_adls = []
    remapped_input_adls = []
    dep_f1_scores = []

    for i in tqdm(range(len(pairs)), desc="Computing metrics"):
        perturbed_doc = perturbed_docs[i]
        original_doc = original_docs[i]
        perturbed_text = perturbed_texts[i]

        perturbed_adls.append(avg_dependency_length(perturbed_doc))
        original_adls.append(avg_dependency_length(original_doc))
        remapped_input_adls.append(remapped_adl(original_doc, perturbed_text))
        dep_f1_scores.append(dep_f1(original_doc, perturbed_doc))

    print(f"\nnum_samples: {len(pairs)}")
    print(f"avg_adl_perturbed: {np.mean(perturbed_adls):.4f}")
    print(f"avg_adl_original: {np.mean(original_adls):.4f}")
    print(f"avg_adl_input_remapped: {np.mean(remapped_input_adls):.4f}")
    print(f"avg_dep_f1_perturbed_vs_original: {np.mean(dep_f1_scores):.4f}")


if __name__ == "__main__":
    main()


# num_samples: 97097
# avg_adl_perturbed: 3.2133
# avg_adl_original: 3.1520
# avg_adl_input_remapped: 3.2189
# avg_dep_f1_perturbed_vs_original: 0.6388

# exact_matches: 5258
# exact_match_rate: 0.0542
#avg_bleu_perturbed_vs_original: 0.4999
