### Min
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_loss_function(loss_type='wmae'):
    ### Get loss function by type
    loss_functions = {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'huber': nn.HuberLoss(),
        'smooth_l1': nn.SmoothL1Loss(),
        'wmae': WeightedMAELoss(),
        'weighted_mse': WeightedMSELoss(),
        'focal': FocalLoss(),
        'combined': CombinedLoss(),
        'uncertainty': UncertaintyLoss(),
        'adaptive': AdaptiveLoss()
    }

    return loss_functions.get(loss_type, WeightedMAELoss())


class WeightedMAELoss(nn.Module):
    ### Weighted Mean Absolute Error Loss - Competition Required

    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            ### Default weights for 5 properties: Tg, FFV, Tc, Density, Rg
            self.weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        else:
            self.weights = torch.tensor(weights)

    def forward(self, pred, target, mask=None):
        if mask is None:
            mask = torch.ones_like(target)

        ### Calculate MAE
        mae = torch.abs(pred - target)

        ### Apply mask
        masked_mae = mae * mask

        ### Apply weights to each property
        weighted_mae = masked_mae * self.weights.to(pred.device)

        ### Calculate weighted average
        total_weight = (mask * self.weights.to(pred.device).unsqueeze(0)).sum()

        if total_weight > 0:
            loss = weighted_mae.sum() / total_weight
        else:
            loss = weighted_mae.sum()

        return loss


class WeightedMSELoss(nn.Module):
    ### Weighted Mean Squared Error Loss

    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            self.weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        else:
            self.weights = torch.tensor(weights)

    def forward(self, pred, target, mask=None):
        if mask is None:
            mask = torch.ones_like(target)

        mse = (pred - target) ** 2
        masked_mse = mse * mask
        weighted_mse = masked_mse * self.weights.to(pred.device).unsqueeze(0)

        total_weight = (mask * self.weights.to(pred.device).unsqueeze(0)).sum()

        if total_weight > 0:
            loss = weighted_mse.sum() / total_weight
        else:
            loss = weighted_mse.sum()

        return loss


class FocalLoss(nn.Module):
    ### Focal Loss for handling hard examples

    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target, mask=None):
        if mask is None:
            mask = torch.ones_like(target)

        mae = torch.abs(pred - target)
        focal_weight = (1 + mae) ** self.gamma
        focal_loss = mae * focal_weight * mask

        total_weight = mask.sum()
        if total_weight > 0:
            loss = focal_loss.sum() / total_weight
        else:
            loss = focal_loss.sum()

        return loss


class CombinedLoss(nn.Module):
    ### Combined MAE and MSE loss

    def __init__(self, alpha=0.7, weights=None):
        super().__init__()
        self.alpha = alpha
        self.mae_loss = WeightedMAELoss(weights)
        self.mse_loss = WeightedMSELoss(weights)

    def forward(self, pred, target, mask=None):
        mae_loss = self.mae_loss(pred, target, mask)
        mse_loss = self.mse_loss(pred, target, mask)

        combined_loss = self.alpha * mae_loss + (1 - self.alpha) * mse_loss
        return combined_loss


class UncertaintyLoss(nn.Module):
    ### Uncertainty-weighted loss for multi-task learning

    def __init__(self, num_tasks=5):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, pred, target, mask=None):
        if mask is None:
            mask = torch.ones_like(target)

        precision = torch.exp(-self.log_vars)
        loss = precision * (pred - target) ** 2 + self.log_vars

        masked_loss = loss * mask
        total_weight = mask.sum()

        if total_weight > 0:
            final_loss = masked_loss.sum() / total_weight
        else:
            final_loss = masked_loss.sum()

        return final_loss


class AdaptiveLoss(nn.Module):
    ### Adaptive loss that switches between MAE and MSE based on error magnitude

    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, target, mask=None):
        if mask is None:
            mask = torch.ones_like(target)

        error = torch.abs(pred - target)

        ### Use MAE for small errors, MSE for large errors
        mae_loss = error * mask
        mse_loss = (error ** 2) * mask

        ### Adaptive weighting
        small_error_mask = (error < self.threshold).float()
        large_error_mask = (error >= self.threshold).float()

        adaptive_loss = small_error_mask * mae_loss + large_error_mask * mse_loss

        total_weight = mask.sum()
        if total_weight > 0:
            loss = adaptive_loss.sum() / total_weight
        else:
            loss = adaptive_loss.sum()

        return loss


class MaskedLossWrapper(nn.Module):
    ### Wrapper for any loss function to handle masking

    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, pred, target, mask=None):
        if mask is None:
            return self.base_loss(pred, target)

        ### Apply mask to both predictions and targets
        masked_pred = pred * mask
        masked_target = target * mask

        return self.base_loss(masked_pred, masked_target)


def create_custom_weights(property_importance):
    ### Create custom weights based on property importance
    if property_importance == 'uniform':
        return [1.0, 1.0, 1.0, 1.0, 1.0]
    elif property_importance == 'density_focused':
        return [1.0, 1.0, 1.0, 2.0, 1.0]  ### Higher weight for Density
    elif property_importance == 'tg_focused':
        return [2.0, 1.0, 1.0, 1.0, 1.0]  ### Higher weight for Tg
    elif property_importance == 'balanced':
        return [1.2, 1.0, 1.0, 1.0, 0.8]  ### Balanced with slight variations
    else:
        return [1.0, 1.0, 1.0, 1.0, 1.0]


def get_loss_with_weights(loss_type='wmae', weights=None, property_importance=None):
    ### Get loss function with custom weights
    if weights is None and property_importance is not None:
        weights = create_custom_weights(property_importance)

    if loss_type == 'wmae':
        return WeightedMAELoss(weights)
    elif loss_type == 'weighted_mse':
        return WeightedMSELoss(weights)
    elif loss_type == 'combined':
        return CombinedLoss(weights=weights)
    else:
        return get_loss_function(loss_type)

### #%#