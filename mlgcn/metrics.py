# -*- coding: utf-8 -*-
"""
Unified multi-label evaluation metrics.

Given ground-truth binarized targets [N, C] in {0,1} and predicted probabilities [N, C],
compute mAP, per-class AP / precision / recall, macro & micro F1 / precision / recall,
and hamming loss.
"""
import numpy as np
from sklearn.metrics import (hamming_loss, average_precision_score, f1_score,
                             precision_score, recall_score)


def compute_metrics(targets, probs, threshold=0.5, class_names=None, eval_name="eval"):
    """
    Args:
        targets:  [N, C] int/float {0,1}
        probs:    [N, C] float probabilities in [0,1]
        threshold: threshold used to binarise predictions for F1 / precision / recall.
        class_names: optional list of C class names (for per-class reporting keys).
        eval_name: prefix used in the returned dict keys.
    Returns: dict of metrics.
    """
    targets = np.asarray(targets)
    probs = np.asarray(probs)
    n_classes = targets.shape[1]
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]

    preds = (probs > threshold).astype(int)

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

        ap = average_precision_score(y_true_c, probs[:, c]) if pos > 0 else 0.0
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
