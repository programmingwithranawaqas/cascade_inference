"""Training-time objectives from the two cascade baselines.

Both are reproduced from the descriptions in the source papers. Where the
published equations are inconsistent with the accompanying prose, the prose is
followed; see the notes on each function.
"""
import torch
import torch.nn.functional as F

LAMBDA_DAR = 0.5
EPSILON_GRID = [0.1, 0.3, 0.5, 0.7]


def logit_norm_loss(logits, labels, kappa):
    """Cross-entropy on logits normalised to unit L2 norm and scaled by 1/kappa.

    The normalisation applies to the loss only; inference uses the ordinary
    softmax of the raw logits.
    """
    norm = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7
    return F.cross_entropy(logits / (norm * kappa), labels)


def difficulty_margin_loss(probs, is_easy, epsilon):
    """Pairwise margin penalty over (easy, difficult) pairs within a batch.

    Easy instances are required to exceed difficult ones in confidence by at
    least `epsilon`. The published pair indicator is stated in a direction that
    contradicts the accompanying text, which asks for the confidence of
    difficult instances to be pushed below that of easy ones; the latter is what
    is implemented.

    Returns zero for batches containing only one difficulty class, which become
    common once the model is accurate. Those batches contribute the task loss
    alone.

    For two classes confidence lies in [0.5, 1], so the pairwise gap lies in
    [-0.5, 0.5] and the hinge is never clipped once epsilon reaches 0.5. Beyond
    that point the loss reduces to (epsilon - gap) and its gradient no longer
    depends on epsilon.
    """
    conf = probs.max(dim=-1).values
    easy = is_easy == 1
    hard = ~easy
    if easy.sum() == 0 or hard.sum() == 0:
        return conf.sum() * 0.0
    gap = conf[easy].unsqueeze(1) - conf[hard].unsqueeze(0)
    return torch.clamp(epsilon - gap, min=0).mean()
