"""
metrics.py — DRISHTI-XAI Unified Evaluation Metrics Module
============================================================
Provides a single interface for all caption generation and
classification evaluation metrics used in the paper.

Caption Metrics:
    - BLEU-1 to BLEU-4  (custom pure-Python, no NLTK dependency)
    - METEOR             (requires: nltk)
    - ROUGE-L            (requires: rouge-score)
    - BERTScore          (requires: bert-score)

Classification Metrics:
    - Per-class Precision, Recall, F1
    - Macro F1
    - Confusion Matrix

Usage (Colab / local):
    from metrics import MetricsSuite
    suite = MetricsSuite(use_bertscore=True, bertscore_lang='bn')
    results = suite.evaluate_captions(references, hypotheses)
    cls_results = suite.evaluate_classification(true_labels, pred_labels)
"""

import math
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────
# 1. BLEU  (pure Python — no extra install)
# ─────────────────────────────────────────────

def _get_ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def compute_bleu_sentence(reference: str, hypothesis: str, max_n: int = 4):
    """
    Sentence-level BLEU-1 to BLEU-4 with Lin & Och (2004) +1 smoothing.
    Returns list [bleu1, bleu2, bleu3, bleu4] as raw fractions (0–1).
    """
    ref_words  = reference.split()
    hyp_words  = hypothesis.split()

    if not hyp_words:
        return [0.0] * max_n

    log_precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = _get_ngrams(hyp_words, n)
        ref_ngrams = _get_ngrams(ref_words, n)

        if not hyp_ngrams:
            log_precisions.append(math.log(1e-4 / (2 ** n)))
            continue

        ref_counts = defaultdict(int)
        for ng in ref_ngrams:
            ref_counts[ng] += 1

        hyp_counts = defaultdict(int)
        for ng in hyp_ngrams:
            hyp_counts[ng] += 1

        clipped = sum(min(cnt, ref_counts[ng]) for ng, cnt in hyp_counts.items())
        # +1 smoothing (avoids log(0) for short sentences)
        precision = (clipped + 1) / (len(hyp_ngrams) + 1)
        log_precisions.append(math.log(precision))

    # Brevity penalty
    bp = 1.0 if len(hyp_words) >= len(ref_words) else \
         math.exp(1 - len(ref_words) / len(hyp_words))

    bleu_scores = []
    for k in range(1, max_n + 1):
        avg_log = sum(log_precisions[:k]) / k
        bleu_scores.append(bp * math.exp(avg_log))

    return bleu_scores   # [b1, b2, b3, b4] each in [0, 1]


def compute_corpus_bleu(references: list, hypotheses: list, max_n: int = 4):
    """
    Corpus-level BLEU: average of sentence-level BLEU scores.
    Returns dict with keys 'bleu1', 'bleu2', 'bleu3', 'bleu4' (×100).
    """
    scores = [compute_bleu_sentence(r, h, max_n)
              for r, h in zip(references, hypotheses)]
    means = [float(np.mean([s[n] for s in scores])) for n in range(max_n)]
    return {f'bleu{n+1}': round(means[n] * 100, 4) for n in range(max_n)}


# ─────────────────────────────────────────────
# 2. METEOR  (requires: nltk)
# ─────────────────────────────────────────────

def _ensure_nltk():
    """Download required NLTK data silently on first call."""
    import nltk
    for pkg in ['wordnet', 'punkt', 'omw-1.4']:
        try:
            nltk.data.find(f'corpora/{pkg}' if pkg != 'punkt' else f'tokenizers/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)


def compute_meteor(reference: str, hypothesis: str) -> float:
    """
    Sentence-level METEOR score using NLTK.
    Returns score in [0, 1].
    NOTE: NLTK METEOR is English-optimized but works as an approximation
          for Bengali subword tokens (a common practice in low-resource NLG).
    """
    try:
        _ensure_nltk()
        from nltk.translate.meteor_score import single_meteor_score
        ref_tokens  = reference.split()
        hyp_tokens  = hypothesis.split()
        return float(single_meteor_score(ref_tokens, hyp_tokens))
    except Exception as e:
        print(f"  [METEOR] Error: {e}. Returning 0.0")
        return 0.0


def compute_corpus_meteor(references: list, hypotheses: list) -> dict:
    """
    Corpus-level METEOR (average of sentence scores × 100).
    Returns dict with key 'meteor'.
    """
    scores = [compute_meteor(r, h) for r, h in zip(references, hypotheses)]
    return {'meteor': round(float(np.mean(scores)) * 100, 4)}


# ─────────────────────────────────────────────
# 3. ROUGE-L  (requires: rouge-score)
# ─────────────────────────────────────────────

class BengaliSpaceTokenizer:
    def tokenize(self, text):
        return text.split()

def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """
    Sentence-level ROUGE-L F1 using google-research rouge-score.
    Returns F1 score in [0, 1].
    Note: use_stemmer=False is correct for Bengali (no stemmer available).
    """
    try:
        from rouge_score import rouge_scorer
        # Use custom tokenizer because default strips Bengali chars
        scorer = rouge_scorer.RougeScorer(
            ['rougeL'], 
            use_stemmer=False,
            tokenizer=BengaliSpaceTokenizer()
        )
        # Ensure non-empty strings to avoid edge-case zero scores
        ref = reference.strip() if reference.strip() else "blank"
        hyp = hypothesis.strip() if hypothesis.strip() else "blank"
        result = scorer.score(ref, hyp)
        return float(result['rougeL'].fmeasure)
    except Exception as e:
        print(f"  [ROUGE-L] Error: {e}. Returning 0.0")
        return 0.0


def compute_corpus_rouge_l(references: list, hypotheses: list) -> dict:
    """
    Corpus-level ROUGE-L (average F1 × 100).
    Returns dict with key 'rouge_l'.
    """
    scores = [compute_rouge_l(r, h) for r, h in zip(references, hypotheses)]
    return {'rouge_l': round(float(np.mean(scores)) * 100, 4)}


# ─────────────────────────────────────────────
# 4. BERTScore  (requires: bert-score)
# ─────────────────────────────────────────────

def compute_bertscore(references: list, hypotheses: list,
                      lang: str = 'bn', verbose: bool = False) -> dict:
    """
    BERTScore Precision, Recall, F1 using multilingual BERT.
    For Bengali ('bn'), uses bert-base-multilingual-cased.

    Args:
        references:  list of reference strings
        hypotheses:  list of hypothesis strings
        lang:        language code ('bn' for Bengali)
        verbose:     if True, prints layer info

    Returns:
        dict with keys 'bertscore_p', 'bertscore_r', 'bertscore_f1' (×100)
    """
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            hypotheses, references,
            lang=lang,
            verbose=verbose,
            rescale_with_baseline=False  # baseline rescaling not available for 'bn'
        )
        return {
            'bertscore_p':  round(float(P.mean().item())  * 100, 4),
            'bertscore_r':  round(float(R.mean().item())  * 100, 4),
            'bertscore_f1': round(float(F1.mean().item()) * 100, 4),
        }
    except Exception as e:
        print(f"  [BERTScore] Error: {e}. Returning 0.0")
        return {'bertscore_p': 0.0, 'bertscore_r': 0.0, 'bertscore_f1': 0.0}


# ─────────────────────────────────────────────
# 5. Classification Metrics  (pure numpy)
# ─────────────────────────────────────────────

LABEL_NAMES = ['Severe_Damage', 'Humanitarian_Rescue', 'Affected_People']


def compute_classification_metrics(true_labels: list,
                                   pred_labels: list,
                                   num_classes: int = 3) -> dict:
    """
    Compute confusion matrix, per-class P/R/F1, and Macro F1.

    Args:
        true_labels: list of ground-truth class IDs (0, 1, 2)
        pred_labels: list of predicted class IDs  (0, 1, 2)
        num_classes: number of classes (default 3)

    Returns:
        dict with 'confusion_matrix', 'precision', 'recall', 'f1',
                  'macro_f1', 'macro_precision', 'macro_recall'
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1

    precision, recall, f1 = [], [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f    = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precision.append(round(float(prec), 4))
        recall.append(round(float(rec),     4))
        f1.append(round(float(f),           4))

    return {
        'confusion_matrix':  cm.tolist(),
        'precision':         precision,
        'recall':            recall,
        'f1':                f1,
        'macro_f1':          round(float(np.mean(f1)),        4),
        'macro_precision':   round(float(np.mean(precision)), 4),
        'macro_recall':      round(float(np.mean(recall)),    4),
    }


# ─────────────────────────────────────────────
# 6. Unified MetricsSuite class
# ─────────────────────────────────────────────

class MetricsSuite:
    """
    Single entry-point for all DRISHTI-XAI evaluation metrics.

    Example (Colab):
        from metrics import MetricsSuite
        suite = MetricsSuite(use_bertscore=True, bertscore_lang='bn')
        cap_results = suite.evaluate_captions(references, hypotheses)
        cls_results = suite.evaluate_classification(true_ids, pred_ids)
        suite.print_report(cap_results, cls_results)
    """

    def __init__(self, use_bertscore: bool = True, bertscore_lang: str = 'bn'):
        self.use_bertscore   = use_bertscore
        self.bertscore_lang  = bertscore_lang

    def evaluate_captions(self, references: list, hypotheses: list) -> dict:
        """
        Run all caption metrics on paired (reference, hypothesis) lists.
        Returns a flat dict of all metric scores.
        """
        results = {}

        # BLEU
        bleu = compute_corpus_bleu(references, hypotheses)
        results.update(bleu)

        # METEOR
        meteor = compute_corpus_meteor(references, hypotheses)
        results.update(meteor)

        # ROUGE-L
        rouge = compute_corpus_rouge_l(references, hypotheses)
        results.update(rouge)

        # BERTScore (optional — requires GPU/longer runtime)
        if self.use_bertscore:
            bscore = compute_bertscore(references, hypotheses,
                                       lang=self.bertscore_lang)
            results.update(bscore)

        return results

    def evaluate_classification(self, true_labels: list,
                                pred_labels: list) -> dict:
        """Run classification metrics."""
        return compute_classification_metrics(true_labels, pred_labels)

    def print_report(self, cap_results: dict = None,
                     cls_results: dict = None):
        """Pretty-print both metric reports (ASCII-safe for all terminals)."""
        sep = "=" * 60

        if cls_results:
            print(f"\n{sep}")
            print("[CLS] TASK 1: DAMAGE CLASSIFICATION METRICS")
            print(sep)
            print(f"  Macro F1        : {cls_results['macro_f1']*100:.2f}%")
            print(f"  Macro Precision : {cls_results['macro_precision']*100:.2f}%")
            print(f"  Macro Recall    : {cls_results['macro_recall']*100:.2f}%")
            print("\n  Per-class breakdown:")
            for i, name in enumerate(LABEL_NAMES):
                p = cls_results['precision'][i]
                r = cls_results['recall'][i]
                f = cls_results['f1'][i]
                print(f"    [{name:<22}] P={p:.4f} R={r:.4f} F1={f:.4f}")
            print("\n  Confusion Matrix (rows=actual, cols=predicted):")
            header = "               " + "  ".join(
                f"{n[:8]:>8}" for n in LABEL_NAMES)
            print(header)
            for i, name in enumerate(LABEL_NAMES):
                row = cls_results['confusion_matrix'][i]
                row_str = (f"  {name[:12]:<12} " +
                           "  ".join(f"{v:>8}" for v in row))
                print(row_str)

        if cap_results:
            print(f"\n{sep}")
            print("[CAP] TASK 2: CAPTION GENERATION METRICS")
            print(sep)
            print(f"  BLEU-1    : {cap_results.get('bleu1',  0.0):.2f}")
            print(f"  BLEU-2    : {cap_results.get('bleu2',  0.0):.2f}")
            print(f"  BLEU-3    : {cap_results.get('bleu3',  0.0):.2f}")
            print(f"  BLEU-4    : {cap_results.get('bleu4',  0.0):.2f}")
            print(f"  METEOR    : {cap_results.get('meteor', 0.0):.2f}")
            print(f"  ROUGE-L   : {cap_results.get('rouge_l',0.0):.2f}")
            if self.use_bertscore:
                print(f"  BERTScore-P  : {cap_results.get('bertscore_p',  0.0):.2f}")
                print(f"  BERTScore-R  : {cap_results.get('bertscore_r',  0.0):.2f}")
                print(f"  BERTScore-F1 : {cap_results.get('bertscore_f1', 0.0):.2f}")
        print(f"{sep}\n")


# ─────────────────────────────────────────────
# 7. Quick smoke-test  (run locally without GPU)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[*] Running metrics smoke-test with dummy Bengali strings...")

    # Dummy Bengali reference-hypothesis pairs
    refs = [
        "বন্যার পানিতে ঘরবাড়ি তলিয়ে গেছে এবং মানুষ আশ্রয়ের সন্ধানে রয়েছে",
        "উদ্ধারকর্মীরা নৌকায় করে ক্ষতিগ্রস্ত মানুষদের সরিয়ে নিচ্ছে",
        "বন্যার পানিতে রাস্তাঘাট ডুবে গেছে যোগাযোগ বিচ্ছিন্ন",
    ]
    hyps = [
        "বন্যার পানিতে ঘরবাড়ি তলিয়ে গেছে",
        "উদ্ধারকর্মীরা নৌকায় মানুষদের সরিয়ে নিচ্ছে",
        "বন্যার পানিতে রাস্তা ডুবে গেছে",
    ]

    # BLEU
    bleu = compute_corpus_bleu(refs, hyps)
    print(f"\n  BLEU scores  : {bleu}")

    # METEOR
    try:
        meteor = compute_corpus_meteor(refs, hyps)
        print(f"  METEOR score : {meteor}")
    except ImportError:
        print("  METEOR       : nltk not installed — pip install nltk")

    # ROUGE-L
    try:
        rouge = compute_corpus_rouge_l(refs, hyps)
        print(f"  ROUGE-L score: {rouge}")
    except ImportError:
        print("  ROUGE-L      : rouge-score not installed — pip install rouge-score")

    # Classification metrics (dummy)
    true_ids = [0, 1, 2, 0, 1]
    pred_ids = [0, 1, 2, 1, 1]
    cls = compute_classification_metrics(true_ids, pred_ids)
    print(f"\n  Macro F1 (dummy cls): {cls['macro_f1']:.4f}")

    # MetricsSuite (no BERTScore for local test — needs GPU)
    suite = MetricsSuite(use_bertscore=False)
    cap_r = suite.evaluate_captions(refs, hyps)
    cls_r = suite.evaluate_classification(true_ids, pred_ids)
    suite.print_report(cap_r, cls_r)
