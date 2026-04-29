
import json
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm

nlp = spacy.load("en_core_web_trf")


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


def analyze_sample(sample):
    input_text = sample["input"]
    prediction = sample["prediction"]
    actual = sample["actual"]

    # Dependency arcs for each
    input_arcs = extract_dependencies(input_text)
    pred_arcs = extract_dependencies(prediction)
    actual_arcs = extract_dependencies(actual)

    # Average dependency lengths
    adl_input = avg_dependency_length(input_arcs)
    adl_pred = avg_dependency_length(pred_arcs)
    adl_actual = avg_dependency_length(actual_arcs)

    # Compare prediction vs actual
    dep_comparison = compare_dependencies(actual, prediction)

    # Compare input vs actual (baseline — how broken is the input?)
    input_comparison = compare_dependencies(actual, input_text)

    return {
        "input": input_text,
        "prediction": prediction,
        "actual": actual,
        "adl_input": round(adl_input, 3),
        "adl_prediction": round(adl_pred, 3),
        "adl_actual": round(adl_actual, 3),
        "pred_vs_actual": dep_comparison,
        "input_vs_actual": input_comparison,
    }


def analyze_dataset(data):
    results = []
    for i, sample in enumerate(tqdm(data)):
        result = analyze_sample(sample)
        result["sample_id"] = i
        results.append(result)

    summary = {
        "num_samples": len(results),
        "avg_adl_input": round(np.mean([r["adl_input"] for r in results]), 3),
        "avg_adl_prediction": round(np.mean([r["adl_prediction"] for r in results]), 3),
        "avg_adl_actual": round(np.mean([r["adl_actual"] for r in results]), 3),
        "avg_dep_f1_pred_vs_actual": round(np.mean([r["pred_vs_actual"]["f1"] for r in results]), 4),
        "avg_dep_f1_input_vs_actual": round(np.mean([r["input_vs_actual"]["f1"] for r in results]), 4),
        "avg_dep_precision": round(np.mean([r["pred_vs_actual"]["precision"] for r in results]), 4),
        "avg_dep_recall": round(np.mean([r["pred_vs_actual"]["recall"] for r in results]), 4),
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
    for i in range(200, 5200, 200):
        print(f"Analyzing Checkpoint {i}")
        input_file = f'/Users/Moham076/Desktop/partial-reverse-100k-bnc_spoken-models/full_samples/full_samples_1k_bnc_spoken_partialReverse_checkpoint-{i}.json'
        output_file = f'bnc_spoken-100k_partialReverse_checkpoint-{i}.csv'

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        results, summary = analyze_dataset(data)
        print(summary)

        # Print summary
        print("=" * 70)
        print("DATASET SUMMARY")
        print("=" * 70)
        for k, v in summary.items():
            print(f"  {k}: {v}")

        # save to CSV
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
        print(f"\nResults saved to {output_file}")