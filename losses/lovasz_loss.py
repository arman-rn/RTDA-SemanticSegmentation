# losses/lovasz_loss.py
"""
This file contains a modern PyTorch implementation of the Lovasz-Softmax loss,
a popular loss function for semantic segmentation that aims to directly optimize
the Jaccard index (Intersection over Union).
"""

from itertools import filterfalse as ifilterfalse

import torch
import torch.nn as nn


def isnan(x):
    """Checks if a tensor value is Not a Number (NaN)."""
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    Calculates the mean of a list or iterator.
    This version is designed to be compatible with generators and can optionally ignore NaN values.
    """
    # Convert the input to an iterator
    l = iter(l)
    # If ignore_nan is True, filter out any NaN values from the iterator
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        # Get the first value and initialize count
        n = 1
        acc = next(l)
    except StopIteration:
        # Handle the case of an empty iterator
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    # Loop through the rest of the values, accumulating the sum and count
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    # Return the final average
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes the gradient of the Lovasz extension with respect to the sorted errors.
    This is the core mathematical component of the loss function, derived from the paper.

    Args:
        gt_sorted (Tensor): A 1D tensor of ground truth labels (0s and 1s) sorted by prediction error.

    Returns:
        Tensor: The gradient of the Lovasz extension.
    """
    p = len(gt_sorted)
    # Calculate the sum of ground truth labels
    gts = gt_sorted.sum()
    # Calculate the intersection at each point in the sorted list
    intersection = gts - torch.cumsum(gt_sorted, dim=0)
    # Calculate the union at each point in the sorted list
    union = gts + torch.cumsum(1.0 - gt_sorted, dim=0)
    # Calculate the Jaccard index (IoU) at each point
    jaccard = 1.0 - intersection / union
    # The gradient is the difference between consecutive Jaccard values
    if p > 1:  # Handle the case of a single-pixel mask
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


# --- Main Loss Class ---


class LovaszSoftmax(nn.Module):
    """
    PyTorch implementation of the Lovasz-Softmax loss for multi-class semantic segmentation.
    This loss is a direct surrogate for optimizing the mean Intersection-over-Union (mIoU) metric.
    """

    def __init__(self, reduction="mean", ignore=None):
        """
        Initializes the loss module.
        Args:
            reduction (str): The reduction method to apply to the final loss. (Not used in this impl, but standard).
            ignore (int, optional): The class index to ignore in the loss calculation (e.g., 255).
        """
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction
        self.ignore = ignore

    def prob_flatten(self, input, target):
        """
        Flattens the predictions and labels to a 2D and 1D tensor respectively.
        This prepares the tensors for per-pixel loss calculation.

        Args:
            input (Tensor): The predicted probabilities from the model, shape [B, C, H, W].
            target (Tensor): The ground truth labels, shape [B, H, W].

        Returns:
            Tuple[Tensor, Tensor]: A tuple of (flattened_probabilities, flattened_labels).
        """
        # Ensure input tensor has 4 dimensions (Batch, Channels, Height, Width)
        if input.dim() != 4:
            raise ValueError(f"Expected 4D input, but got {input.dim()}D")

        num_class = input.size(1)

        # Reshape the input tensor from [B, C, H, W] to [B*H*W, C]
        # .permute changes the order of dimensions to [B, H, W, C]
        # .contiguous() ensures the tensor is stored in a contiguous block of memory
        # .view(-1, num_class) flattens the first three dimensions
        input_flatten = input.permute(0, 2, 3, 1).contiguous().view(-1, num_class)

        # Flatten the target tensor from [B, H, W] to [B*H*W]
        target_flatten = target.view(-1)

        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, probas, labels, classes="present"):
        """
        Calculates the Lovasz-Softmax loss for a flattened list of predictions.

        Args:
            probas (Tensor): Flattened class probabilities, shape [P, C] where P is the number of pixels.
            labels (Tensor): Flattened ground truth labels, shape [P].
            classes (str): 'present' to average loss only over classes present in the labels.

        Returns:
            Tensor: The calculated scalar loss value.
        """
        # If there are no pixels to calculate loss on, return 0
        if probas.numel() == 0:
            return probas * 0.0

        C = probas.size(1)  # Number of classes
        losses = []  # List to store the loss for each class

        # Determine which classes to calculate loss for. 'present' means only for classes in this batch's labels.
        class_to_sum = list(range(C)) if classes == "all" else torch.unique(labels)

        # Loop through each class ID to calculate its specific Lovasz loss
        for c in class_to_sum:
            # Create a "foreground" mask for the current class 'c'. It's 1 where the label is 'c', and 0 otherwise.
            fg = (labels == c).float()

            # --- This is the crucial block for handling the ignore_index ---
            if self.ignore is not None:
                # 1. First, if the class we are currently processing IS the ignore_index itself, we skip it.
                if c == self.ignore:
                    continue

                # 2. Next, we filter out all ignored pixels from our calculation for this valid class.
                # 'valid_mask' is True for all pixels that are NOT the ignore_index.
                valid_mask = labels != self.ignore
                # Apply this mask to the foreground and probabilities. This ensures ignored pixels
                # don't contribute to the loss of any class.
                probas_c = probas[valid_mask, c]
                fg = fg[valid_mask]
            else:
                # If no ignore_index is set, just get the probabilities for the current class.
                probas_c = probas[:, c]

            # If there are no ground truth pixels for this class in the batch, skip it.
            if fg.sum() == 0:
                continue

            # Calculate the prediction errors for this class (difference between ground truth and prediction)
            errors = (fg - probas_c).abs()
            # Sort the errors in descending order. This is a key step in the Lovasz-Softmax algorithm.
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            # Get the permutation indices to sort the ground truth labels in the same order as the errors.
            perm = perm.data
            fg_sorted = fg[perm]

            # Calculate the final loss for this class by taking the dot product of the sorted errors
            # and the Lovasz gradient. This is the "Lovasz extension".
            losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

        # Average the losses from all present classes to get the final loss for the batch.
        return mean(losses)

    def forward(self, probas, labels):
        """
        The main forward pass for the loss function.

        Args:
            probas (Tensor): The output from the model after a Softmax layer. Shape [B, C, H, W].
            labels (Tensor): The ground truth segmentation map. Shape [B, H, W].

        Returns:
            Tensor: The final computed scalar loss.
        """
        # First, flatten the predictions and labels.
        probas, labels = self.prob_flatten(probas, labels)
        # Then, calculate the flat Lovasz-Softmax loss.
        loss = self.lovasz_softmax_flat(probas, labels, classes="present")
        return loss
