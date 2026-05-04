import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.plot import COLORS, setup_acl_style

PERTURBATIONS = [
    "partialReverse",
    "localShuffle5",
    "localShuffle3",
    "localShuffle",
    "fullShuffle",
    "wordHop",
]

METRIC_LABELS = {
    "exact_match": "Exact Match",
    "BLEU": "BLEU",
    "dep_f1": "Dependency F1",
    "avg_dependency_length": "Average Dependency Length",
}


def checkpoint_sort_key(checkpoint_name):
    if checkpoint_name == "final":
        return (1, float("inf"))

    if checkpoint_name.startswith("checkpoint-"):
        try:
            return (0, int(checkpoint_name.split("-", 1)[1]))
        except ValueError:
            return (0, checkpoint_name)

    return (0, checkpoint_name)


def parse_metric_filename(file_path, metric_name):
    stem = Path(file_path).stem
    prefix = "results_"
    suffix = f"_{metric_name}"

    if not stem.startswith(prefix) or not stem.endswith(suffix):
        raise ValueError(f"Unsupported metric filename: {file_path}")

    body = stem[len(prefix):-len(suffix)]

    for perturbation in sorted(PERTURBATIONS, key=len, reverse=True):
        marker = f"_{perturbation}"
        if body.endswith(marker):
            dataset = body[:-len(marker)]
            if not dataset:
                raise ValueError(f"Missing dataset name in filename: {file_path}")
            return dataset, perturbation

    raise ValueError(
        f"Could not determine perturbation for {file_path}. "
        f"Expected one of: {', '.join(PERTURBATIONS)}"
    )


def find_metric_files(inputs, metric_name):
    files = []
    expected_name = f"results_*_{metric_name}.json"

    for input_path in inputs:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")

        if path.is_file():
            if path.match(expected_name):
                files.append(path)
            continue

        direct_matches = sorted(path.rglob(expected_name))
        if direct_matches:
            files.extend(direct_matches)
            continue

        for run_dir in sorted(child for child in path.iterdir() if child.is_dir()):
            evaluation_dir = run_dir / "evaluation"
            if evaluation_dir.is_dir():
                files.extend(sorted(evaluation_dir.rglob(expected_name)))

    if not files:
        raise FileNotFoundError(
            f"No files matching {expected_name} found in: {', '.join(inputs)}"
        )

    return sorted(set(files))


def load_metric_value(file_path, checkpoint="final"):
    with Path(file_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, (int, float)):
        return float(data)

    if checkpoint in data:
        return float(data[checkpoint])

    items = sorted(data.items(), key=lambda item: checkpoint_sort_key(item[0]))
    if not items:
        raise ValueError(f"No checkpoint values found in {file_path}")

    return float(items[-1][1])


def compute_error(values, error_bar):
    if len(values) <= 1:
        return 0.0

    if error_bar == "std":
        return float(np.std(values, ddof=1))

    if error_bar == "sem":
        return float(np.std(values, ddof=1) / np.sqrt(len(values)))

    raise ValueError(f"Unsupported error-bar type: {error_bar}")


def build_metric_groups(files, metric_name, checkpoint):
    grouped_scores = defaultdict(list)

    for file_path in files:
        dataset, perturbation = parse_metric_filename(file_path, metric_name)
        value = load_metric_value(file_path, checkpoint=checkpoint)
        grouped_scores[(dataset, perturbation)].append(value)

    return grouped_scores


def format_dataset_label(label):
    return label.replace("_", " ")


def format_perturbation_label(label):
    mapping = {
        "partialReverse": "PartialReverse",
        "localShuffle": "LocalShuffle",
        "localShuffle3": "LocalShuffle3",
        "localShuffle5": "LocalShuffle5",
        "fullShuffle": "FullShuffle",
        "wordHop": "WordHop",
    }
    return mapping.get(label, label)


def plot_seed_bar_chart(grouped_scores, metric_name, output_file, error_bar="std", title=""):
    datasets = sorted({dataset for dataset, _ in grouped_scores})
    perturbations = [
        perturbation
        for perturbation in PERTURBATIONS
        if any(current == perturbation for _, current in grouped_scores)
    ]

    if not datasets or not perturbations:
        raise ValueError("No grouped scores available to plot.")

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    x_positions = np.arange(len(datasets))
    total_bar_width = 0.8
    bar_width = total_bar_width / max(len(perturbations), 1)

    for index, perturbation in enumerate(perturbations):
        means = []
        errors = []

        for dataset in datasets:
            values = grouped_scores.get((dataset, perturbation), [])
            if values:
                means.append(float(np.mean(values)))
                errors.append(compute_error(values, error_bar))
            else:
                means.append(np.nan)
                errors.append(0.0)

        offset = (index - (len(perturbations) - 1) / 2.0) * bar_width
        ax.bar(
            x_positions + offset,
            means,
            width=bar_width * 0.92,
            color=COLORS[index % len(COLORS)],
            edgecolor="white",
            linewidth=0.6,
            yerr=errors,
            capsize=3,
            ecolor="0.25",
            alpha=0.9,
            label=format_perturbation_label(perturbation),
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([format_dataset_label(label) for label in datasets], rotation=20, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name))

    if metric_name in {"exact_match", "BLEU", "dep_f1"}:
        ax.set_ylim(0, 1.0)
    else:
        all_values = [value for values in grouped_scores.values() for value in values]
        ymax = max(all_values) if all_values else 1.0
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)

    if title:
        ax.set_title(title)

    if len(perturbations) > 1:
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_file, dpi=400, bbox_inches="tight", facecolor="white")

    output_path = Path(output_file)
    if output_path.suffix.lower() == ".png":
        fig.savefig(
            output_path.with_suffix(".pdf"),
            dpi=400,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create a bar plot with seed error bars from evaluation result JSON files."
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        required=True,
        help="Result directories or result JSON files from different seeds.",
    )
    parser.add_argument(
        "-m",
        "--metric",
        required=True,
        choices=sorted(METRIC_LABELS),
        help="Metric to plot.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output PNG file path. A PDF is also saved automatically when using .png.",
    )
    parser.add_argument(
        "--checkpoint",
        default="final",
        help="Checkpoint key to read from the results JSON. Defaults to final.",
    )
    parser.add_argument(
        "--error-bar",
        default="std",
        choices=["std", "sem"],
        help="Error bar type across seeds.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional figure title.",
    )
    args = parser.parse_args()

    setup_acl_style()

    metric_files = find_metric_files(args.inputs, args.metric)
    grouped_scores = build_metric_groups(
        metric_files,
        metric_name=args.metric,
        checkpoint=args.checkpoint,
    )

    for (dataset, perturbation), values in sorted(grouped_scores.items()):
        print(
            f"{dataset} / {perturbation}: "
            f"n={len(values)}, mean={np.mean(values):.4f}, {args.error_bar}={compute_error(values, args.error_bar):.4f}"
        )

    output_file = args.output or f"{args.metric}_seed_barplot.png"
    plot_seed_bar_chart(
        grouped_scores,
        metric_name=args.metric,
        output_file=output_file,
        error_bar=args.error_bar,
        title=args.title,
    )
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    main()
