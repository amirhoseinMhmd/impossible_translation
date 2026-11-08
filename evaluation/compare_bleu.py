import argparse
import json

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm


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
    smoothing = SmoothingFunction().method1

    scores = []
    for perturbed, original in tqdm(pairs, desc="Computing BLEU"):
        perturbed_tokens = perturbed.split()
        original_tokens = original.split()
        score = sentence_bleu([original_tokens], perturbed_tokens, smoothing_function=smoothing)
        scores.append(score)

    print(f"num_samples: {len(pairs)}")
    print(f"avg_bleu_perturbed_vs_original: {sum(scores) / len(scores):.4f}")


if __name__ == "__main__":
    main()
