import os
import numpy as np
import matplotlib.pyplot as plt
from DataPreparation import DataPreparation
from LogisticRegression import LogisticRegressionModel

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_CONFIGS = {
    "Unigram":              {"ngram_range": (1, 1)},
    "Bigram":               {"ngram_range": (1, 2)},
    "Unigram + Positional": {"ngram_range": (1, 1), "use_positional_features": True},
    "Bigram + Positional":  {"ngram_range": (1, 2), "use_positional_features": True}
}

METRICS = ["precision", "recall", "f1-score"]


def collect_results(dataset):
  results = {name: [] for name in MODEL_CONFIGS}
  for fold, data in dataset.items():
    print(f"Fold {fold}: train={len(data['train'])}  test={len(data['test'])}")
    for name, cfg in MODEL_CONFIGS.items():
      model = LogisticRegressionModel(**cfg)
      model.fit(data["train"])
      results[name].append(model.predict(data["test"], output_as_dict=True))
  return results


def summarize(results):
  summary = {}
  for name, reports in results.items():
    fold_vals = {m: [r["macro avg"][m] for r in reports] for m in METRICS}
    fold_vals["accuracy"] = [r["accuracy"] for r in reports]
    summary[name] = {
        m: (np.mean(v), np.std(v)) for m, v in fold_vals.items()
    }
  return summary


def save_summary_table(summary):
  fig, ax = plt.subplots(figsize=(10, 2.2))
  ax.axis("off")
  cols = ["Model"] + [m.title() for m in METRICS] + ["Accuracy"]
  rows = [
      [name] + [f"{summary[name][m][0]:.3f} \u00b1 {summary[name][m][1]:.3f}"
                for m in METRICS + ["accuracy"]]
      for name in summary
  ]
  tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
  tbl.auto_set_font_size(False)
  tbl.set_fontsize(10)
  tbl.scale(1.2, 1.6)
  for j in range(len(cols)):
    tbl[0, j].set_facecolor("#4472C4")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
  fig.savefig(f"{RESULTS_DIR}/summary_table.png", dpi=200, bbox_inches="tight")
  plt.close(fig)


def save_macro_avg_chart(summary):
  labels = METRICS + ["accuracy"]
  x = np.arange(len(labels))
  w = 0.25
  fig, ax = plt.subplots(figsize=(9, 5))
  for i, name in enumerate(summary):
    means = [summary[name][m][0] for m in labels]
    stds = [summary[name][m][1] for m in labels]
    ax.bar(x + i * w, means, w, yerr=stds, label=name, capsize=3)
  ax.set_xticks(x + w)
  ax.set_xticklabels([m.title() for m in labels])
  ax.set_ylabel("Score")
  ax.set_title("Macro-Avg Metrics Across Folds (mean \u00b1 std)")
  ax.set_ylim(0, 1.05)
  ax.legend()
  fig.tight_layout()
  fig.savefig(f"{RESULTS_DIR}/macro_avg_bar_chart.png",
              dpi=200, bbox_inches="tight")
  plt.close(fig)


def save_per_class_f1_chart(results):
  skip = {"accuracy", "macro avg", "weighted avg"}
  classes = [k for k in results[list(results)[0]][0] if k not in skip]
  x = np.arange(len(classes))
  w = 0.25
  fig, ax = plt.subplots(figsize=(10, 5))
  for i, name in enumerate(results):
    means, stds = [], []
    for cls in classes:
      f1s = [r[cls]["f1-score"] for r in results[name]]
      means.append(np.mean(f1s))
      stds.append(np.std(f1s))
    ax.bar(x + i * w, means, w, yerr=stds, label=name, capsize=3)
  ax.set_xticks(x + w)
  ax.set_xticklabels(classes, rotation=30, ha="right")
  ax.set_ylabel("F1-Score")
  ax.set_title("Per-Class F1 Across Folds (mean \u00b1 std)")
  ax.set_ylim(0, 1.05)
  ax.legend()
  fig.tight_layout()
  fig.savefig(f"{RESULTS_DIR}/per_class_f1_bar_chart.png",
              dpi=200, bbox_inches="tight")
  plt.close(fig)


if __name__ == "__main__":
  data_prep = DataPreparation("data/mentalmanip_detailed.csv")
  data_prep.load_data()
  dataset = data_prep.cross_validation_split(k=5)

  results = collect_results(dataset)
  summary = summarize(results)

  save_summary_table(summary)
  save_macro_avg_chart(summary)
  save_per_class_f1_chart(results)

  print(f"Results saved to {RESULTS_DIR}/")
