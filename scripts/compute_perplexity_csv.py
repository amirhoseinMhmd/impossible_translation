"""
Compute GPT-2 perplexity for input/prediction/actual columns in fullShuffle CSV files.

Reads evaluation_results/{corpus}/{perturbation}/{seed}/full_samples_*.csv,
averages perplexity across seeds per checkpoint, and prints results
ready to paste into perplexity_plot2.py.

Usage:
    python3 scripts/compute_perplexity_csv.py --perturbation fullShuffle
    python3 scripts/compute_perplexity_csv.py --perturbation fullShuffle --corpus gutenberg
    python3 scripts/compute_perplexity_csv.py --perturbation fullShuffle --checkpoint final
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------
def calculate_perplexity(text, model, tokenizer, max_length=512):
    if not text or not str(text).strip():
        return None
    encodings = tokenizer(str(text), return_tensors="pt",
                          truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    ppl = torch.exp(loss).item()
    return ppl if np.isfinite(ppl) else None


def perplexity_for_column(texts, model, tokenizer, desc=""):
    ppls = []
    for text in tqdm(texts, desc=desc, leave=False):
        p = calculate_perplexity(text, model, tokenizer)
        if p is not None:
            ppls.append(p)
    return ppls


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def find_csv_files(results_dir, corpus_filter, perturbation):
    """Return {(corpus, seed): [(ckpt_num, path), ...]}"""
    results_dir = Path(results_dir)
    grouped = defaultdict(list)

    for corpus_dir in sorted(results_dir.iterdir()):
        if not corpus_dir.is_dir():
            continue
        corpus = corpus_dir.name
        if corpus_filter and corpus != corpus_filter:
            continue
        pert_dir = corpus_dir / perturbation
        if not pert_dir.is_dir():
            continue
        for seed_dir in sorted(pert_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            seed = seed_dir.name
            for csv_file in sorted(seed_dir.glob("full_samples_*.csv")):
                m = re.search(r"checkpoint-(\d+)", csv_file.name)
                if m:
                    ckpt = int(m.group(1))
                    grouped[(corpus, seed)].append((ckpt, csv_file))

    return grouped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbation", default="fullShuffle")
    parser.add_argument("--corpus", default=None, help="Filter to one corpus")
    parser.add_argument("--results-dir", default="evaluation_results")
    parser.add_argument("--checkpoint", default=None,
                        help="Only process this checkpoint number, or 'last'")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    # Discover files
    grouped = find_csv_files(args.results_dir, args.corpus, args.perturbation)
    if not grouped:
        print(f"No CSV files found for perturbation={args.perturbation}")
        return

    print(f"\nFound {len(grouped)} (corpus, seed) combos:")
    for (corpus, seed), files in sorted(grouped.items()):
        print(f"  {corpus}/{args.perturbation}/{seed}: {len(files)} checkpoints")

    # Load model
    print(f"\nLoading {args.model}...")
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    model.eval()
    print("Model ready.\n")

    # {corpus: {ckpt: {col: [seed_means]}}}
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for (corpus, seed), ckpt_files in sorted(grouped.items()):
        ckpt_files_sorted = sorted(ckpt_files, key=lambda x: x[0])

        # Filter checkpoints
        if args.checkpoint == "last":
            ckpt_files_sorted = [ckpt_files_sorted[-1]]
        elif args.checkpoint is not None:
            ckpt_files_sorted = [(c, p) for c, p in ckpt_files_sorted
                                 if c == int(args.checkpoint)]

        if not ckpt_files_sorted:
            continue

        # input & actual don't change across checkpoints — compute once from the first CSV
        first_csv = ckpt_files_sorted[0][1]
        print(f"  [{corpus}/{seed}] computing input/actual once from {first_csv.name}")
        df_first = pd.read_csv(first_csv)

        static_ppls = {}
        for col in ["input", "actual"]:
            if col in df_first.columns:
                ppls = perplexity_for_column(df_first[col].dropna().tolist(),
                                             model, tokenizer, desc=f"    {col}")
                if ppls:
                    static_ppls[col] = float(np.mean(ppls))
                    print(f"    {col}: mean={static_ppls[col]:.2f} (n={len(ppls)})")

        # prediction changes per checkpoint — compute for each
        for ckpt_num, csv_path in ckpt_files_sorted:
            print(f"  [{corpus}/{seed}] checkpoint-{ckpt_num}: prediction")
            df = pd.read_csv(csv_path)

            # Store static values once per checkpoint slot
            for col, mean_val in static_ppls.items():
                results[corpus][ckpt_num][col].append(mean_val)

            if "prediction" in df.columns:
                ppls = perplexity_for_column(df["prediction"].dropna().tolist(),
                                             model, tokenizer, desc="    prediction")
                if ppls:
                    seed_mean = float(np.mean(ppls))
                    results[corpus][ckpt_num]["prediction"].append(seed_mean)
                    print(f"    prediction: mean={seed_mean:.2f} (n={len(ppls)})")

    # Aggregate across seeds and save + plot
    import json
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    output_dir = Path(args.results_dir).parent / "perplexity_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS (mean ± std across seeds)")
    print("=" * 70)

    for corpus, ckpt_data in sorted(results.items()):
        print(f"\nCorpus: {corpus}  |  Perturbation: {args.perturbation}")
        print(f"{'Checkpoint':>12}  {'Input':>10}  {'Pred Mean':>12}  {'Pred Std':>10}  {'Actual':>10}  {'Seeds':>6}")
        print("-" * 72)

        checkpoints, pred_means, pred_stds = [], [], []
        inp_ref = act_ref = None

        for ckpt in sorted(ckpt_data.keys()):
            cols = ckpt_data[ckpt]
            inp  = float(np.mean(cols["input"]))   if cols.get("input")      else float("nan")
            act  = float(np.mean(cols["actual"]))  if cols.get("actual")     else float("nan")
            preds = cols.get("prediction", [])
            pmean = float(np.mean(preds))          if preds else float("nan")
            pstd  = float(np.std(preds, ddof=1))   if len(preds) > 1 else 0.0
            n = len(preds)

            print(f"{ckpt:>12}  {inp:>10.2f}  {pmean:>12.2f}  {pstd:>10.2f}  {act:>10.2f}  {n:>6}")
            checkpoints.append(ckpt)
            pred_means.append(pmean)
            pred_stds.append(pstd)
            if inp_ref is None:
                inp_ref = inp
            if act_ref is None:
                act_ref = act

        # Save raw per-seed results to JSON
        json_path = output_dir / f"{corpus}_{args.perturbation}_perplexity.json"
        save_data = {
            str(ckpt): {
                col: ckpt_data[ckpt][col]
                for col in ["input", "prediction", "actual"]
                if ckpt_data[ckpt].get(col)
            }
            for ckpt in sorted(ckpt_data.keys())
        }
        with open(json_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\n  Saved raw results → {json_path}")

        # Overall average (for bar chart reference)
        all_inp  = [v for ck in ckpt_data.values() for v in ck.get("input",      [])]
        all_pred = [v for ck in ckpt_data.values() for v in ck.get("prediction", [])]
        all_act  = [v for ck in ckpt_data.values() for v in ck.get("actual",     [])]
        print(f"\n  Overall avg — input: {np.mean(all_inp):.2f}  "
              f"prediction: {np.mean(all_pred):.2f}  actual: {np.mean(all_act):.2f}")
        print(f"""
# ── paste into perplexity_plot2.py ──
    "FullShuffle": {{
        "input":      {np.mean(all_inp):.2f},
        "prediction": {np.mean(all_pred):.2f},
    }},
""")

        # Plot: prediction perplexity vs checkpoint with std error bars
        fig, ax = plt.subplots(figsize=(6, 4))

        checkpoints = np.array(checkpoints)
        pred_means  = np.array(pred_means)
        pred_stds   = np.array(pred_stds)

        ax.plot(checkpoints, pred_means,
                color="#D55E00", marker="o", linewidth=1.8, markersize=5,
                label="Prediction (mean across seeds)")
        ax.errorbar(checkpoints, pred_means, yerr=pred_stds,
                    fmt="none", ecolor="#D55E00", elinewidth=1.2,
                    capsize=3, capthick=0.8, alpha=0.7)

        if inp_ref is not None:
            ax.axhline(inp_ref, color="#0072B2", linestyle="--", linewidth=1.2,
                       label=f"Input ({inp_ref:.0f})")
        if act_ref is not None:
            ax.axhline(act_ref, color="#222222", linestyle=":", linewidth=1.2,
                       label=f"Actual ({act_ref:.0f})")

        ax.set_yscale("log")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Geometric Mean Perplexity")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, which="major", alpha=0.3, linewidth=0.4)
        fig.tight_layout()

        plot_path = output_dir / f"{corpus}_{args.perturbation}_perplexity.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        fig.savefig(plot_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot → {plot_path}")


if __name__ == "__main__":
    main()
