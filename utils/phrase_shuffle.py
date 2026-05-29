import argparse
import os
from typing import List, Optional

import benepar
import spacy
from tqdm import tqdm
from numpy.random import default_rng

from utils import load_sentences_from_file, save_dataset

SHUFFLE_LABELS = {"NP", "VP", "ADJP", "ADVP", "PP", "PRT", "QP", "FRAG"}

try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    nlp = spacy.load("en_core_web_md")

if not nlp.has_pipe("benepar"):
    try:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    except LookupError as exc:
        raise RuntimeError(
            "benepar is installed, but the 'benepar_en3' model is missing. "
            "Install it with: python -m benepar.download benepar_en3"
        ) from exc


def _leaf_phrases(sent):
    """Return spans of leaf-level phrases (no shuffleable children)."""
    spans = []

    def recurse(tree):
        label = tree._.labels[0] if tree._.labels else None
        children = list(tree._.children)
        child_labels = {
            c._.labels[0] for c in children
            if c._.labels and c._.labels[0] in SHUFFLE_LABELS
        }
        if label in SHUFFLE_LABELS and not child_labels:
            if tree.end - tree.start > 1:
                spans.append((tree.start, tree.end))
        else:
            for child in children:
                recurse(child)

    recurse(sent)
    return spans


def phrase_shuffle(text: str, seed: int = None) -> Optional[str]:
    text = text.strip()
    if len(text.split()) < 3:
        return None

    try:
        doc = nlp(text)
    except Exception:
        return None

    rng = default_rng(seed)
    tokens = list(doc)
    result = tokens[:]

    for sent in doc.sents:
        try:
            for start, end in _leaf_phrases(sent):
                phrase = result[start:end]
                if len(phrase) < 2:
                    continue
                shuffled = phrase[:]
                rng.shuffle(shuffled)
                result[start:end] = shuffled
        except Exception:
            continue

    output = []
    for i, tok in enumerate(result):
        output.append(tok.text)
        if tokens[i].whitespace_:
            output.append(" ")

    return "".join(output).strip()


def phrase_shuffle_batch(texts: List[str], seed: int = None) -> List[tuple]:
    training_data = []
    for sentence in tqdm(texts):
        perturbed = phrase_shuffle(sentence, seed=seed)
        if perturbed is not None:
            training_data.append((perturbed, sentence))
    return training_data


def pre_process(input_file, training_data_path, seed=42):
    sentences = load_sentences_from_file(input_file)
    training_data = phrase_shuffle_batch(sentences, seed=seed)
    save_dataset(training_data, training_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = f"training_data_{os.path.basename(args.input).split('.')[0]}_phraseShuffer.json"
    pre_process(args.input, output, seed=args.seed)
