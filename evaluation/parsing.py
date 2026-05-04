
import json
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    nlp = spacy.load("en_core_web_sm")

# Special tokens inserted by perturbation functions — excluded from token mapping
SPECIAL_TOKENS = {"R", "S", "P"}


def extract_dependencies(text):
    doc = nlp(text)
    arcs = []
    for token in doc:
        arcs.append({
            "dependent": token.text,
            "dep_idx": token.i,
            "head": token.head.text,
            "head_idx": token.head.i,
            "dep_rel": token.dep_,
            "arc_length": abs(token.i - token.head.i)
        })
    return arcs


def avg_dependency_length(arcs):
    lengths = [a["arc_length"] for a in arcs if a["dep_rel"] != "ROOT"]
    return np.mean(lengths) if lengths else 0.0


def get_dep_triples(text):
    """Returns a set of (dependent, dep_rel, head) triples."""
    doc = nlp(text)
    triples = set()
    for token in doc:
        if token.dep_ != "ROOT":
            triples.add((token.text.lower(), token.dep_, token.head.text.lower()))
    return triples


def compare_dependencies(actual_text, prediction_text):
    """
    Compares dependency triples between actual and prediction.
    Returns precision, recall, F1 of recovered dependency arcs.
    """
    actual_triples = get_dep_triples(actual_text)
    pred_triples = get_dep_triples(prediction_text)

    if not actual_triples or not pred_triples:
        return {"precision": 0, "recall": 0, "f1": 0,
                "matched": 0, "actual_count": len(actual_triples),
                "pred_count": len(pred_triples)}

    matched = actual_triples & pred_triples
    precision = len(matched) / len(pred_triples) if pred_triples else 0
    recall = len(matched) / len(actual_triples) if actual_triples else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched": len(matched),
        "actual_count": len(actual_triples),
        "pred_count": len(pred_triples)
    }


def _build_token_mapping(actual_tokens, perturbed_tokens):
    """
    Builds a mapping from original token position to perturbed token position.
    Matches greedily left-to-right by token text, ignoring special tokens.
    Returns dict: {original_idx: perturbed_idx}
    """
    clean_perturbed = [
        (i, t) for i, t in enumerate(perturbed_tokens)
        if t not in SPECIAL_TOKENS
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


def compute_perturbed_adl(actual_text, perturbed_text):
    """
    Computes ADL of the perturbed string by remapping the original parse arcs
    to their new positions. Valid for pure reordering perturbations
    (LocalShuffle, PartialReverse). For WordHOP, returns None.
    """
    doc = nlp(actual_text)
    actual_tokens = [t.text for t in doc]
    perturbed_tokens = perturbed_text.split()

    mapping = _build_token_mapping(actual_tokens, perturbed_tokens)

    lengths = []
    for token in doc:
        if token.dep_ == "ROOT":
            continue
        orig_dep = token.i
        orig_head = token.head.i
        if orig_dep in mapping and orig_head in mapping:
            lengths.append(abs(mapping[orig_dep] - mapping[orig_head]))

    return round(np.mean(lengths), 3) if lengths else 0.0


def compute_perturbed_baseline_f1(actual_text, perturbed_text):
    """
    Computes baseline Triple F1 for the perturbed input without parsing
    the scrambled text. A dependency triple (dep, rel, head) from the original
    parse is considered preserved in the perturbed string if both tokens are
    present and their relative order (dep before/after head) is unchanged.
    Returns precision, recall, F1 against the original triple set.
    """
    doc = nlp(actual_text)
    actual_tokens = [t.text for t in doc]
    perturbed_tokens = perturbed_text.split()

    mapping = _build_token_mapping(actual_tokens, perturbed_tokens)

    actual_triples = set()
    preserved_triples = set()

    for token in doc:
        if token.dep_ == "ROOT":
            continue
        triple = (token.text.lower(), token.dep_, token.head.text.lower())
        actual_triples.add(triple)

        orig_dep = token.i
        orig_head = token.head.i
        if orig_dep in mapping and orig_head in mapping:
            # Triple is preserved if relative order of dep and head is unchanged
            orig_order = orig_dep < orig_head
            pert_order = mapping[orig_dep] < mapping[orig_head]
            if orig_order == pert_order:
                preserved_triples.add(triple)

    if not actual_triples:
        return {"precision": 0, "recall": 0, "f1": 0,
                "matched": 0, "actual_count": 0, "pred_count": 0}

    matched = actual_triples & preserved_triples
    precision = len(matched) / len(preserved_triples) if preserved_triples else 0
    recall = len(matched) / len(actual_triples)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched": len(matched),
        "actual_count": len(actual_triples),
        "pred_count": len(preserved_triples),
    }


def analyze_sample(sample, wordhop=False):
    """
    wordhop=True: skips remapping-based input metrics since WordHOP alters
    token vocabulary (lemmatization), making position mapping unreliable.
    """
    input_text = sample["input"]
    prediction = sample["prediction"]
    actual = sample["actual"]

    pred_arcs = extract_dependencies(prediction)
    actual_arcs = extract_dependencies(actual)

    adl_pred = avg_dependency_length(pred_arcs)
    adl_actual = avg_dependency_length(actual_arcs)

    if wordhop:
        # WordHOP alters verb forms — token mapping is unreliable.
        # Report None for input-based metrics.
        adl_input = None
        input_comparison = {"precision": None, "recall": None, "f1": None,
                            "matched": None, "actual_count": None, "pred_count": None}
    else:
        adl_input = compute_perturbed_adl(actual, input_text)
        input_comparison = compute_perturbed_baseline_f1(actual, input_text)

    dep_comparison = compare_dependencies(actual, prediction)

    return {
        "input": input_text,
        "prediction": prediction,
        "actual": actual,
        "adl_input": adl_input,
        "adl_prediction": round(adl_pred, 3),
        "adl_actual": round(adl_actual, 3),
        "pred_vs_actual": dep_comparison,
        "input_vs_actual": input_comparison,
    }


def analyze_dataset(data, wordhop=False):
    results = []
    for i, sample in enumerate(tqdm(data)):
        result = analyze_sample(sample, wordhop=wordhop)
        result["sample_id"] = i
        results.append(result)

    def safe_mean(values):
        valid = [v for v in values if v is not None]
        return round(np.mean(valid), 4) if valid else None

    summary = {
        "num_samples": len(results),
        "avg_adl_input": safe_mean([r["adl_input"] for r in results]),
        "avg_adl_prediction": safe_mean([r["adl_prediction"] for r in results]),
        "avg_adl_actual": safe_mean([r["adl_actual"] for r in results]),
        "avg_dep_f1_pred_vs_actual": safe_mean([r["pred_vs_actual"]["f1"] for r in results]),
        "avg_dep_f1_input_vs_actual": safe_mean([r["input_vs_actual"]["f1"] for r in results]),
        "avg_dep_precision": safe_mean([r["pred_vs_actual"]["precision"] for r in results]),
        "avg_dep_recall": safe_mean([r["pred_vs_actual"]["recall"] for r in results]),
    }

    return results, summary


def print_sample_analysis(result):
    print(f"{'='*70}")
    print(f"Input:      {result['input']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Actual:     {result['actual']}")
    print(f"\nAvg Dependency Length:")
    print(f"  Input:      {result['adl_input']}")
    print(f"  Prediction: {result['adl_prediction']}")
    print(f"  Actual:     {result['adl_actual']}")
    print(f"\nDependency Recovery (Prediction vs Actual):")
    d = result['pred_vs_actual']
    print(f"  Precision: {d['precision']}  Recall: {d['recall']}  F1: {d['f1']}")
    print(f"  Matched: {d['matched']} / {d['actual_count']} actual arcs")
    print(f"\nDependency Recovery (Input vs Actual) [baseline]:")
    d2 = result['input_vs_actual']
    print(f"  Precision: {d2['precision']}  Recall: {d2['recall']}  F1: {d2['f1']}")
    print(f"  Matched: {d2['matched']} / {d2['actual_count']} actual arcs")


if __name__ == "__main__":
    import argparse
    import glob as glob_module
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        default="experiments/full-samples")
    parser.add_argument("--output_dir",
                        default="dep-results")
    parser.add_argument("--perturbation", choices=["localShuffle", "partialReverse", "wordHop", "all"],
                        default="all")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    perturbations = ["localShuffle", "partialReverse", "wordHop"] \
        if args.perturbation == "all" else [args.perturbation]

    for perturbation in perturbations:
        wordhop = perturbation == "wordHop"
        pattern = os.path.join(args.input_dir, f"*_{perturbation}_*.json")
        files = sorted(glob_module.glob(pattern))

        if not files:
            print(f"No files found for {perturbation}, skipping.")
            continue

        print(f"\n{'='*70}")
        print(f"Perturbation: {perturbation}  ({len(files)} files)")
        print(f"{'='*70}")

        for input_file in files:
            basename = os.path.basename(input_file).replace(".json", "")
            output_file = os.path.join(args.output_dir, f"{basename}.csv")

            print(f"\nProcessing: {basename}")
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            results, summary = analyze_dataset(data, wordhop=wordhop)

            for k, v in summary.items():
                print(f"  {k}: {v}")

            rows = []
            for r in results:
                rows.append({
                    "sample_id": r["sample_id"],
                    "input": r["input"],
                    "prediction": r["prediction"],
                    "actual": r["actual"],
                    "adl_input": r["adl_input"],
                    "adl_prediction": r["adl_prediction"],
                    "adl_actual": r["adl_actual"],
                    "dep_precision": r["pred_vs_actual"]["precision"],
                    "dep_recall": r["pred_vs_actual"]["recall"],
                    "dep_f1": r["pred_vs_actual"]["f1"],
                    "baseline_f1": r["input_vs_actual"]["f1"],
                })
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
            print(f"  Saved → {output_file}")