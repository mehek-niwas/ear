import torch
from transformers import AutoModelForSequenceClassification
from collections import namedtuple


def compute_negative_entropy(
    inputs: tuple, attention_mask: torch.torch, return_values: bool = False
):
    """Compute the negative entropy across layers of a network for given inputs.
    Args:
        - input: tuple. Tuple of length num_layers. Each item should be in the form: BHSS
        - attention_mask. Tensor with dim: BS
    """

    inputs = torch.stack(inputs)  #  LayersBatchHeadsSeqlenSeqlen

    assert inputs.ndim == 5, "Here we expect 5 dimensions in the form LBHSS"

    #  average over attention heads
    pool_heads = inputs.mean(2)

    batch_size = pool_heads.shape[1]
    samples_entropy = list()
    neg_entropies = list()
    for b in range(batch_size):
        #  get inputs from non-padded tokens of the current sample
        mask = attention_mask[b]
        sample = pool_heads[:, b, mask.bool(), :]
        sample = sample[:, :, mask.bool()]

        #  get the negative entropy for each non-padded token
        neg_entropy = (sample.softmax(-1) * sample.log_softmax(-1)).sum(-1)
        if return_values:
            neg_entropies.append(neg_entropy.detach())

        #  get the "average entropy" that traverses the layer
        mean_entropy = neg_entropy.mean(-1)

        #  store the sum across all the layers
        samples_entropy.append(mean_entropy.sum(0))

    # average over the batch
    final_entropy = torch.stack(samples_entropy).mean()
    if return_values:
        return final_entropy, neg_entropies
        # entropies.shape = (layers, nonpadded token)
    else:
        return final_entropy


def compute_mehek_negative_entropy(
    inputs: tuple,
    attention_mask: torch.Tensor,
    return_values: bool = False,
    return_heads: bool = False,
):
    """Compute the negative entropy across layers and attention heads for given inputs.

    Args:
        - inputs: tuple. Tuple of length num_layers. Each item should be in the form: BHSS
        - attention_mask: Tensor with dim: BS
        - return_values: bool. If True, return per-sample entropies for further inspection.
        - return_heads: bool. If True, returns entropy per head; otherwise averages over heads.

    Returns:
        - final_entropy: Average negative entropy across the batch.
        - neg_entropies (optional): List of tensors (one per sample) with shape:
            - (Layers, Heads, Tokens) if return_heads=True
            - (Layers, Tokens) if return_heads=False
    """

    inputs = torch.stack(inputs)  # Layers x Batch x Heads x SeqLen x SeqLen
    assert inputs.ndim == 5, (
        "Expected 5 dimensions in the form (Layers, Batch, Heads, SeqLen, SeqLen)"
    )

    batch_size = inputs.shape[1]
    samples_entropy = []
    neg_entropies = []

    for b in range(batch_size):
        # Get inputs from non-padded tokens of the current sample
        mask = attention_mask[b]
        sample = inputs[
            :, b, :, mask.bool(), :
        ]  # Layers x Heads x NonPaddedTokens x SeqLen
        sample = sample[
            :, :, :, mask.bool()
        ]  # Layers x Heads x NonPaddedTokens x NonPaddedTokens

        # Calculate negative entropy per head for each token
        neg_entropy = (sample.softmax(-1) * sample.log_softmax(-1)).sum(
            -1
        )  # Layers x Heads x NonPaddedTokens

        # Average over heads if return_heads is False
        if not return_heads:
            neg_entropy = neg_entropy.mean(1)  # Layers x NonPaddedTokens

        if return_values:
            neg_entropies.append(neg_entropy.detach())

        # Average over tokens and sum across layers to get batch-level entropy
        mean_entropy = neg_entropy.mean(
            -1
        )  # Layers (if not return_heads) or Layers x Heads
        samples_entropy.append(
            mean_entropy.sum(0)
        )  # Scalar (if not return_heads) or Heads

    # Average across the batch
    final_entropy = torch.stack(
        samples_entropy
    ).mean()  # Scalar (if not return_heads) or Heads

    if return_values:
        return final_entropy, neg_entropies
    else:
        return final_entropy


EARClassificationOutput = namedtuple(
    "EARClassificationOutput", ["model_output", "negative_entropy", "reg_loss", "loss"]
)


class EARModelForSequenceClassification(torch.nn.Module):
    def __init__(
        self, model_name_or_path, ear_reg_strength: float = 0.01, model_kwargs={}
    ):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, **model_kwargs
        )
        self.ear_reg_strength = ear_reg_strength

    def forward(self, **model_kwargs):
        output = self.model(**model_kwargs, output_attentions=True)

        negative_entropy = compute_negative_entropy(
            output.attentions, model_kwargs["attention_mask"]
        )
        reg_loss = self.ear_reg_strength * negative_entropy
        loss = reg_loss + output.loss

        return EARClassificationOutput(
            model_output=output,
            negative_entropy=negative_entropy,
            reg_loss=reg_loss,
            loss=loss,
        )

    def save_pretrained(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)
