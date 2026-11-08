import argparse
import json


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

    matches = sum(1 for perturbed, original in pairs if perturbed == original)
    total = len(pairs)
    accuracy = matches / total if total else 0.0

    print(f"num_samples: {total}")
    print(f"exact_matches: {matches}")
    print(f"exact_match_rate: {accuracy:.4f}")


if __name__ == "__main__":
    main()
