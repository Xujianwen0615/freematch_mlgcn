# -*- coding: utf-8 -*-
"""
Unified multi-label evaluation metrics.

Given ground-truth binarized targets [N, C] in {0,1} and predicted probabilities or raw
logits [N, C], compute mAP, per-class AP / precision / recall, macro & micro F1 / precision /
recall, and hamming loss.
"""
import numpy as np
from sklearn.metrics import (hamming_loss, average_precision_score, f1_score,
                             precision_score, recall_score)


def compute_logit_stats(logits):
    """Return compact diagnostics for detecting logit explosion/saturation."""
    logits = np.asarray(logits, dtype=np.float64)
    finite = np.isfinite(logits)
    finite_values = logits[finite]
    stats = {
        "nonfinite": int((~finite).sum()),
        "abs_gt_16_ratio": float(np.mean(np.abs(finite_values) > 16.0)) if finite_values.size else 0.0,
        "abs_gt_80_ratio": float(np.mean(np.abs(finite_values) > 80.0)) if finite_values.size else 0.0,
    }
    if finite_values.size:
        stats.update({
            "min": float(finite_values.min()),
            "max": float(finite_values.max()),
            "mean": float(finite_values.mean()),
            "std": float(finite_values.std()),
        })
    else:
        stats.update({
            "min": float("nan"), "max": float("nan"),
            "mean": float("nan"), "std": float("nan"),
        })
    return stats


def compute_metrics(targets, probs, threshold=0.5, class_names=None, eval_name="eval",
                    from_logits=False):
    """
    Args:
        targets:  [N, C] int/float {0,1}
        probs:    [N, C] probabilities in [0,1], or raw logits when
                  ``from_logits=True``. The historical parameter name is
                  retained for API compatibility.
        threshold: threshold used to binarise predictions for F1 / precision / recall.
        class_names: optional list of C class names (for per-class reporting keys).
        eval_name: prefix used in the returned dict keys.
        from_logits: calculate AP directly from raw logits and convert the
                     probability threshold to its logit decision boundary.
    Returns: dict of metrics.
    """
    targets = np.asarray(targets)
    scores = np.asarray(probs, dtype=np.float64)
    if targets.ndim != 2 or scores.ndim != 2 or targets.shape != scores.shape:
        raise ValueError(
            f"targets/scores must have the same [N, C] shape; "
            f"got {targets.shape} and {scores.shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN/Inf; inspect the model logits before evaluation")
    n_classes = targets.shape[1]
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]

    if from_logits:
        if not 0.0 < threshold < 1.0:
            raise ValueError("probability threshold must be strictly between 0 and 1")
        decision_threshold = float(np.log(threshold) - np.log1p(-threshold))
    else:
        decision_threshold = threshold
    preds = (scores > decision_threshold).astype(int)

    # ---------- mAP (mean average precision, ignoring classes with no positives) ----------
    ap_list = []
    per_class_ap = []
    per_class_precision = []
    per_class_recall = []
    per_class_pos = []
    for c in range(n_classes):
        y_true_c = targets[:, c]
        y_pred_c = preds[:, c]
        pos = int(y_true_c.sum())
        per_class_pos.append(pos)

        # AP depends on ranking only. Raw logits preserve that ranking even
        # when sigmoid would round large scores to identical 0/1 values.
        ap = average_precision_score(y_true_c, scores[:, c]) if pos > 0 else 0.0
        per_class_ap.append(ap)
        if pos > 0:
            ap_list.append(ap)

        tp = int((y_true_c * y_pred_c).sum())
        fp = int(((1 - y_true_c) * y_pred_c).sum())
        fn = int((y_true_c * (1 - y_pred_c)).sum())
        per_class_precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        per_class_recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    mAP = float(np.mean(ap_list)) if ap_list else 0.0

    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    macro_prec = float(precision_score(targets, preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(targets, preds, average="macro", zero_division=0))
    micro_f1 = float(f1_score(targets, preds, average="micro", zero_division=0))
    micro_prec = float(precision_score(targets, preds, average="micro", zero_division=0))
    micro_rec = float(recall_score(targets, preds, average="micro", zero_division=0))
    ham = float(hamming_loss(targets, preds))

    metrics = {
        f"{eval_name}/mAP": mAP,
        f"{eval_name}/macro-F1": macro_f1,
        f"{eval_name}/macro-precision": macro_prec,
        f"{eval_name}/macro-recall": macro_rec,
        f"{eval_name}/micro-F1": micro_f1,
        f"{eval_name}/micro-precision": micro_prec,
        f"{eval_name}/micro-recall": micro_rec,
        f"{eval_name}/hamming_loss": ham,
    }

    # Per-class AP (sorted, for readable reporting).
    for c in range(n_classes):
        metrics[f"{eval_name}/per_class/{class_names[c]}/AP"] = float(per_class_ap[c])
        metrics[f"{eval_name}/per_class/{class_names[c]}/precision"] = float(per_class_precision[c])
        metrics[f"{eval_name}/per_class/{class_names[c]}/recall"] = float(per_class_recall[c])
        metrics[f"{eval_name}/per_class/{class_names[c]}/pos_count"] = int(per_class_pos[c])

    return metrics


def format_report(metrics, eval_name="eval", top=None):
    """Return a human-readable multi-line report string from a metrics dict."""
    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"  Multi-label report ({eval_name})")
    lines.append(f"{'='*80}")
    lines.append(f"  mAP={metrics.get(f'{eval_name}/mAP', 0.0):.4f}   "
                 f"Macro-F1={metrics.get(f'{eval_name}/macro-F1', 0.0):.4f}   "
                 f"Micro-F1={metrics.get(f'{eval_name}/micro-F1', 0.0):.4f}   "
                 f"Hamming={metrics.get(f'{eval_name}/hamming_loss', 0.0):.4f}")
    lines.append(f"  Macro-P/R: {metrics.get(f'{eval_name}/macro-precision', 0.0):.4f} / "
                 f"{metrics.get(f'{eval_name}/macro-recall', 0.0):.4f}   "
                 f"Micro-P/R: {metrics.get(f'{eval_name}/micro-precision', 0.0):.4f} / "
                 f"{metrics.get(f'{eval_name}/micro-recall', 0.0):.4f}")
    lines.append(f"{'-'*80}")
    lines.append(f"  {'Class':<28}{'#Pos':>6}{'AP':>9}{'Prec':>9}{'Rec':>9}")
    per_keys = sorted([k for k in metrics if f"{eval_name}/per_class/" in k and k.endswith("/AP")])
    rows = []
    for k in per_keys:
        name = k.split("/")[-2]
        pos = metrics.get(f"{eval_name}/per_class/{name}/pos_count", 0)
        ap = metrics[k]
        prec = metrics.get(f"{eval_name}/per_class/{name}/precision", 0.0)
        rec = metrics.get(f"{eval_name}/per_class/{name}/recall", 0.0)
        rows.append((name, pos, ap, prec, rec))
    rows.sort(key=lambda r: r[2], reverse=True)
    for name, pos, ap, prec, rec in rows:
        lines.append(f"  {name[:28]:<28}{pos:>6}{ap:>9.4f}{prec:>9.4f}{rec:>9.4f}")
    lines.append(f"{'='*80}")
    return "\n".join(lines)
