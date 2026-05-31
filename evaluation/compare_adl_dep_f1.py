import argparse
import json
from pathlib import Path

import numpy as np
import spacy
from accelerate.utils import tqdm

try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    nlp = spacy.load("en_core_web_sm")


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
    args = parser.parse_args()

    pairs = load_pairs(args.input)

    perturbed_adls = []
    original_adls = []
    remapped_input_adls = []
    dep_f1_scores = []

    for perturbed, original in tqdm(pairs):
        perturbed_doc = nlp(perturbed)
        original_doc = nlp(original)

        perturbed_adls.append(avg_dependency_length(perturbed_doc))
        original_adls.append(avg_dependency_length(original_doc))
        remapped_input_adls.append(remapped_adl(original_doc, perturbed))
        dep_f1_scores.append(dep_f1(original_doc, perturbed_doc))

    print(f"num_samples: {len(pairs)}")
    print(f"avg_adl_perturbed: {np.mean(perturbed_adls):.4f}")
    print(f"avg_adl_original: {np.mean(original_adls):.4f}")
    print(f"avg_adl_input_remapped: {np.mean(remapped_input_adls):.4f}")
    print(f"avg_dep_f1_perturbed_vs_original: {np.mean(dep_f1_scores):.4f}")


if __name__ == "__main__":
    main()