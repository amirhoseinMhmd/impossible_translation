import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Allow running `python evaluation/evaluate.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.bleu import bleu_score
from evaluation.exact_match import exact_match
from evaluation.parsing import analyze_dataset

PERTURBATIONS = [
    "partialReverse",
    "localShuffle5",
    "localShuffle3",
    "localShuffle",
    "fullShuffle",
    "wordHop",
]

DETAIL_COLUMNS = [
    "sample_id",
    "input",
    "prediction",
    "actual",
    "adl_input",
    "adl_prediction",
    "adl_actual",
    "dep_precision",
    "dep_recall",
    "dep_f1",
    "baseline_f1",
]


def format_metric(value):
    return "n/a" if value is None else f"{value:.4f}"


def save_json(data, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def save_csv(rows, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_full_samples_filename(file_path):
    stem = Path(file_path).stem
    if not stem.startswith("full_samples_"):
        raise ValueError(f"Unsupported full-samples filename: {file_path}")

    body = stem[len("full_samples_"):]
    match = re.match(r"(?P<prefix>.+)_(?P<checkpoint>checkpoint-\d+|final)$", body)
    if not match:
        raise ValueError(
            "Could not parse dataset / perturbation / checkpoint from "
            f"filename: {file_path}"
        )

    prefix = match.group("prefix")
    checkpoint = match.group("checkpoint")

    for perturbation in sorted(PERTURBATIONS, key=len, reverse=True):
        suffix = f"_{perturbation}"
        if prefix.endswith(suffix):
            dataset = prefix[: -len(suffix)]
            if not dataset:
                raise ValueError(f"Missing dataset name in filename: {file_path}")
            return {
                "dataset": dataset,
                "perturbation": perturbation,
                "checkpoint": checkpoint,
                "basename": stem,
            }

    raise ValueError(
        f"Could not determine perturbation from filename: {file_path}. "
        f"Expected one of: {', '.join(PERTURBATIONS)}"
    )


def checkpoint_sort_key(checkpoint_name):
    if checkpoint_name == "final":
        return (1, float("inf"))

    match = re.match(r"checkpoint-(\d+)$", checkpoint_name)
    if match:
        return (0, int(match.group(1)))

    return (0, checkpoint_name)


def build_detail_rows(analysis_results):
    rows = []
    for result in analysis_results:
        rows.append({
            "sample_id": result["sample_id"],
            "input": result["input"],
            "prediction": result["prediction"],
            "actual": result["actual"],
            "adl_input": result["adl_input"],
            "adl_prediction": result["adl_prediction"],
            "adl_actual": result["adl_actual"],
            "dep_precision": result["pred_vs_actual"]["precision"],
            "dep_recall": result["pred_vs_actual"]["recall"],
            "dep_f1": result["pred_vs_actual"]["f1"],
            "baseline_f1": result["input_vs_actual"]["f1"],
        })
    return rows


def evaluate_full_samples_file(file_path, output_dir):
    metadata = parse_full_samples_filename(file_path)

    with Path(file_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    predictions = [sample["prediction"] for sample in data]
    actuals = [sample["actual"] for sample in data]

    exact_match_score = exact_match(predictions, actuals)
    bleu = bleu_score(predictions, actuals)
    analysis_results, dependency_summary = analyze_dataset(
        data,
        wordhop=metadata["perturbation"] == "wordHop",
    )

    detail_rows = build_detail_rows(analysis_results)
    csv_output = Path(output_dir) / f"{metadata['basename']}.csv"
    save_csv(detail_rows, csv_output)

    summary = {
        "file": str(Path(file_path).resolve()),
        "dataset": metadata["dataset"],
        "perturbation": metadata["perturbation"],
        "checkpoint": metadata["checkpoint"],
        "num_samples": len(data),
        "exact_match": exact_match_score,
        "BLEU": bleu,
        "avg_dependency_length": dependency_summary["avg_adl_prediction"],
        "dep_f1": dependency_summary["avg_dep_f1_pred_vs_actual"],
        **dependency_summary,
    }

    summary_output = Path(output_dir) / f"{metadata['basename']}_metrics.json"
    save_json(summary, summary_output)

    print(
        f"Evaluated {metadata['basename']}: "
        f"EM={format_metric(exact_match_score)}, BLEU={format_metric(bleu)}, "
        f"ADL={format_metric(summary['avg_dependency_length'])}, "
        f"dep_F1={format_metric(summary['dep_f1'])}"
    )

    return summary


def find_full_sample_files(input_dir, perturbation="all", dataset=None):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = []
    for file_path in input_path.rglob("full_samples_*.json"):
        metadata = parse_full_samples_filename(file_path)
        if perturbation != "all" and metadata["perturbation"] != perturbation:
            continue
        if dataset and metadata["dataset"] != dataset:
            continue
        files.append(file_path)

    return sorted(files, key=lambda path: (
        parse_full_samples_filename(path)["dataset"],
        parse_full_samples_filename(path)["perturbation"],
        checkpoint_sort_key(parse_full_samples_filename(path)["checkpoint"]),
    ))


def save_group_results(group_summaries, output_dir):
    grouped = defaultdict(list)
    for summary in group_summaries:
        grouped[(summary["dataset"], summary["perturbation"])].append(summary)

    for (dataset, perturbation), summaries in grouped.items():
        summaries.sort(key=lambda item: checkpoint_sort_key(item["checkpoint"]))

        exact_match_results = {}
        bleu_results = {}
        avg_dependency_length_results = {}
        dep_f1_results = {}
        summary_results = {}

        for summary in summaries:
            checkpoint = summary["checkpoint"]
            if checkpoint in summary_results:
                raise ValueError(
                    f"Duplicate checkpoint '{checkpoint}' found for "
                    f"{dataset}/{perturbation}. Evaluate one run folder at a time "
                    "or rename the artifacts to disambiguate them."
                )

            exact_match_results[checkpoint] = summary["exact_match"]
            bleu_results[checkpoint] = summary["BLEU"]
            avg_dependency_length_results[checkpoint] = summary["avg_dependency_length"]
            dep_f1_results[checkpoint] = summary["dep_f1"]
            summary_results[checkpoint] = summary

        prefix = f"results_{dataset}_{perturbation}"
        save_json(exact_match_results, Path(output_dir) / f"{prefix}_exact_match.json")
        save_json(bleu_results, Path(output_dir) / f"{prefix}_BLEU.json")
        save_json(
            avg_dependency_length_results,
            Path(output_dir) / f"{prefix}_avg_dependency_length.json",
        )
        save_json(dep_f1_results, Path(output_dir) / f"{prefix}_dep_f1.json")
        save_json(summary_results, Path(output_dir) / f"{prefix}_summary.json")

        print(f"\nSaved aggregated results for {dataset} / {perturbation}")


def evaluate_checkpoint_folder(input_dir, output_dir, perturbation="all", dataset=None):
    files = find_full_sample_files(
        input_dir=input_dir,
        perturbation=perturbation,
        dataset=dataset,
    )

    if not files:
        raise FileNotFoundError(
            f"No full-sample checkpoint files found in {input_dir} "
            f"for perturbation={perturbation!r} dataset={dataset!r}."
        )

    print(f"Found {len(files)} full-sample files to evaluate.")
    summaries = [evaluate_full_samples_file(file_path, output_dir) for file_path in files]
    save_group_results(summaries, output_dir)
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved full-sample checkpoint files and save per-checkpoint metrics."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing full_samples_*.json files.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="evaluation/checkpoint-results",
        help="Directory where CSVs and JSON summaries will be saved.",
    )
    parser.add_argument(
        "-t",
        "--perturbation",
        type=str,
        default="all",
        choices=PERTURBATIONS + ["all"],
        help="Filter evaluation to one perturbation type.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset name filter parsed from the full_samples filename.",
    )
    args = parser.parse_args()

    evaluate_checkpoint_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        perturbation=args.perturbation,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
