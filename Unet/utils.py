import torch
from torch import Tensor
import torch.nn.functional as F


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    # sum dim (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, batch_first: bool = True, epsilon: float = 1e-6):
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), batch_first, epsilon)


def dice_loss(mask_pred: Tensor, mask: Tensor):
    input = F.softmax(mask_pred, dim=1).float()
    target = F.one_hot(mask, 2).permute(0, 3, 1, 2).float()
    # input.shape == target.shape == [bs,n,w,h]
    return 1 - multiclass_dice_coeff(input, target)
