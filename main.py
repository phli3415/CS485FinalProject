import os
import numpy as np
import matplotlib.pyplot as plt
from DataPreparation import DataPreparation
from LogisticRegression import LogisticRegressionModel

import pandas as pd

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

NGRAM_FAMILY = {
    "Unigram": {"ngram_range": (1, 1)},
    "Bigram":  {"ngram_range": (1, 2)},
    "Trigram": {"ngram_range": (1, 3)},
}

POSITIONAL_FAMILY = {
    "Unigram + Positional": {"ngram_range": (1, 1), "use_positional_features": True},
    "Bigram + Positional":  {"ngram_range": (1, 2), "use_positional_features": True},
    "Trigram + Positional": {"ngram_range": (1, 3), "use_positional_features": True},
}

POS_FAMILY = {
    "Unigram + POS": {"ngram_range": (1, 1), "use_pos_features": True},
    "Bigram + POS":  {"ngram_range": (1, 2), "use_pos_features": True},
    "Trigram + POS": {"ngram_range": (1, 3), "use_pos_features": True},
}

COMBINED_FAMILY = {
    "Unigram + Positional + POS": {"ngram_range": (1, 1), "use_positional_features": True, "use_pos_features": True},
    "Bigram + Positional + POS":  {"ngram_range": (1, 2), "use_positional_features": True, "use_pos_features": True},
    "Trigram + Positional + POS": {"ngram_range": (1, 3), "use_positional_features": True, "use_pos_features": True},
}

METRICS = ["precision", "recall", "f1-score"]


def collect_results(dataset, model_configs, model_class=LogisticRegressionModel, coef_capture_names=None):
  """Train each model on each fold, return one classification_report dict
  per (model, fold). coef_capture_names is a SET of model names whose
  coefficients should be captured"""
  if coef_capture_names is None:
    coef_capture_names = set()
  results = {name: [] for name in model_configs}
  coef_records = []

  for fold, data in dataset.items():
    print(f"Fold {fold}: train={len(data['train'])}  test={len(data['test'])}")
    for name, cfg in model_configs.items():
      model = model_class(**cfg)
      model.fit(data["train"])
      results[name].append(model.predict(data["test"], output_as_dict=True))
      if name in coef_capture_names:
        summary = model.feature_summary()
        for feat, coef in summary["dense_pos_coefs"].items():
          coef_records.append({
            "model": name, "fold": fold, "feature": feat,
            "coef": coef, "positive_class": summary["positive_class"],
          })
  return results, coef_records


def summarize(results):
  summary = {}
  for name, reports in results.items():
    fold_vals = {m: [r["macro avg"][m] for r in reports] for m in METRICS}
    fold_vals["accuracy"] = [r["accuracy"] for r in reports]
    classes = [k for k in reports[0] if k not in {"accuracy", "macro avg", "weighted avg"}]
    for cls in classes:
      fold_vals[f"{cls}_f1"] = [r[cls]["f1-score"] for r in reports]
    summary[name] = {m: (np.mean(v), np.std(v)) for m, v in fold_vals.items()}
  return summary


def pick_champion(family_summary, criterion_key="manipulation_f1"):
  """Return the model name with highest mean on the chosen criterion.
  Default is manipulation-class F1, since the goal of the application is
  catching manipulation when it occurs (not just overall fairness)."""
  return max(family_summary, key=lambda name: family_summary[name][criterion_key][0])


def save_summary_table(summary, filename="summary_table.png", figsize=(10, 2.2), title=None, champion=None):
  """Makes the summary table. If `title` is given, draws it above the table.
  If `champion` is given, that row's first column gets a star prefix."""
  fig, ax = plt.subplots(figsize=figsize)
  ax.axis("off")
  cols = ["Model"] + [m.title() for m in METRICS] + ["Accuracy"]
  rows = []
  for name in summary:
    label = f"\u2605 {name}" if name == champion else name
    rows.append(
        [label] + [f"{summary[name][m][0]:.3f} \u00b1 {summary[name][m][1]:.3f}"
                   for m in METRICS + ["accuracy"]]
    )
  tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
  tbl.auto_set_font_size(False)
  tbl.set_fontsize(10)
  tbl.scale(1.2, 1.6)
  for j in range(len(cols)):
    tbl[0, j].set_facecolor("#4472C4")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
  if champion is not None:
    for r_idx, name in enumerate(summary, start=1):
      if name == champion:
        for j in range(len(cols)):
          tbl[r_idx, j].set_text_props(fontweight="bold")
        break
  if title:
    ax.set_title(title, fontweight="bold", pad=14)
  fig.savefig(f"{RESULTS_DIR}/{filename}", dpi=200, bbox_inches="tight")
  plt.close(fig)


def save_macro_avg_chart(summary, filename="macro_avg_bar_chart.png", figsize=(9, 5), title="Macro-Avg Metrics Across Folds (mean \u00b1 std)"):
  labels = METRICS + ["accuracy"]
  x = np.arange(len(labels))
  w = 0.8 / max(len(summary), 1)
  fig, ax = plt.subplots(figsize=figsize)
  for i, name in enumerate(summary):
    means = [summary[name][m][0] for m in labels]
    stds = [summary[name][m][1] for m in labels]
    ax.bar(x + i * w, means, w, yerr=stds, label=name, capsize=3)
  ax.set_xticks(x + w * (len(summary) - 1) / 2)
  ax.set_xticklabels([m.title() for m in labels])
  ax.set_ylabel("Score")
  ax.set_title(title)
  ax.set_ylim(0, 1.05)
  ax.legend(fontsize=9, loc="upper right")
  fig.tight_layout()
  fig.savefig(f"{RESULTS_DIR}/{filename}", dpi=200, bbox_inches="tight")
  plt.close(fig)


def save_per_class_f1_chart(results, filename="per_class_f1_bar_chart.png", figsize=(10, 5), title="Per-Class F1 Across Folds (mean \u00b1 std)"):
  skip = {"accuracy", "macro avg", "weighted avg"}
  classes = [k for k in results[list(results)[0]][0] if k not in skip]
  x = np.arange(len(classes))
  w = 0.8 / max(len(results), 1)
  fig, ax = plt.subplots(figsize=figsize)
  for i, name in enumerate(results):
    means, stds = [], []
    for cls in classes:
      f1s = [r[cls]["f1-score"] for r in results[name]]
      means.append(np.mean(f1s))
      stds.append(np.std(f1s))
    ax.bar(x + i * w, means, w, yerr=stds, label=name, capsize=3)
  ax.set_xticks(x + w * (len(results) - 1) / 2)
  ax.set_xticklabels(classes, rotation=30, ha="right")
  ax.set_ylabel("F1-Score")
  ax.set_title(title)
  ax.set_ylim(0, 1.05)
  ax.legend(fontsize=9, loc="upper right")
  fig.tight_layout()
  fig.savefig(f"{RESULTS_DIR}/{filename}", dpi=200, bbox_inches="tight")
  plt.close(fig)


def save_coef_plot(coef_records, model_name, filename="pos_coef_plot.png"):
  """Filter coef_records to one model and plot its mean coefficients."""
  filtered = [r for r in coef_records if r["model"] == model_name]
  if not filtered:
    print(f"  (no coef records for {model_name} -- skipping coef plot)")
    return
  coef_df = pd.DataFrame(filtered)
  pos_class = coef_df["positive_class"].iloc[0]
  agg = (coef_df.groupby("feature")["coef"].agg(["mean", "std"]).reset_index())
  agg["abs_mean"] = agg["mean"].abs()
  agg = agg.sort_values("abs_mean", ascending=True)

  colors = ["#C00000" if m > 0 else "#4472C4" for m in agg["mean"]]
  fig, ax = plt.subplots(figsize=(9, 7))
  y = np.arange(len(agg))
  ax.barh(y, agg["mean"], xerr=agg["std"], color=colors, capsize=3, alpha=0.9)
  ax.set_yticks(y)
  ax.set_yticklabels(agg["feature"])
  ax.axvline(0, color="black", linewidth=0.8)
  ax.set_xlabel(f"Coefficient  (positive \u2192 pushes toward '{pos_class}')")
  ax.set_title(f"{model_name}: Dense POS Coefficients (mean \u00b1 std, 5-fold CV)")
  ax.grid(axis="x", alpha=0.3)
  fig.tight_layout()
  fig.savefig(f"{RESULTS_DIR}/{filename}", dpi=200, bbox_inches="tight")
  plt.close(fig)


if __name__ == "__main__":
  data_prep = DataPreparation("data/mentalmanip_detailed.csv")
  data_prep.load_data(include_pos=True, include_word_pos=True, strip_speakers=True)
  dataset = data_prep.cross_validation_split(k=5)

  ngram_results, _ = collect_results(dataset, NGRAM_FAMILY)
  positional_results, _ = collect_results(dataset, POSITIONAL_FAMILY)
  pos_results, pos_coef_records = collect_results(dataset, POS_FAMILY, coef_capture_names=set(POS_FAMILY.keys()), )
  combined_results, combined_coef_records = collect_results(dataset, COMBINED_FAMILY, coef_capture_names=set(COMBINED_FAMILY.keys()), )

  ngram_summary = summarize(ngram_results)
  positional_summary = summarize(positional_results)
  pos_summary = summarize(pos_results)
  combined_summary = summarize(combined_results)



  ngram_champion = pick_champion(ngram_summary)
  positional_champion = pick_champion(positional_summary)
  pos_champion = pick_champion(pos_summary)
  combined_champion = pick_champion(combined_summary)

  print(f"\nChampions (by manipulation-class F1):")
  print(f"N-gram family: {ngram_champion}")
  print(f"Positional family: {positional_champion}")
  print(f"POS family: {pos_champion}")
  print(f"Combined family: {combined_champion}")

  save_summary_table(ngram_summary, filename="family_table_ngram.png", title="N-gram Family", champion=ngram_champion)
  save_summary_table(positional_summary, filename="family_table_positional.png", title="Positional Family", champion=positional_champion)
  save_summary_table(pos_summary, filename="family_table_pos.png", title="POS Family", champion=pos_champion)
  save_summary_table(combined_summary, filename="family_table_combined.png", title="Combined Family", champion=combined_champion)

  champion_results = {
    ngram_champion: ngram_results[ngram_champion],
    positional_champion: positional_results[positional_champion],
    pos_champion: pos_results[pos_champion],
    combined_champion: combined_results[combined_champion],
  }
  champion_summary = {
    ngram_champion: ngram_summary[ngram_champion],
    positional_champion: positional_summary[positional_champion],
    pos_champion: pos_summary[pos_champion],
    combined_champion: combined_summary[combined_champion],
  }

  save_summary_table(champion_summary, filename="summary_table.png", figsize=(11, 2.6), title="Champion Comparison: Best Model From Each Family")
  save_macro_avg_chart(champion_summary, filename="macro_avg_bar_chart.png", title="Champion Comparison: Macro-Avg Metrics (mean \u00b1 std)")
  save_per_class_f1_chart(champion_results, filename="per_class_f1_bar_chart.png", title="Champion Comparison: Per-Class F1 (mean \u00b1 std)")

  save_coef_plot(combined_coef_records, model_name=combined_champion, filename="combined_coef_plot.png")
  save_coef_plot(pos_coef_records, model_name=pos_champion, filename="pos_coef_plot.png")

  print(f"\nResults saved to {RESULTS_DIR}/")