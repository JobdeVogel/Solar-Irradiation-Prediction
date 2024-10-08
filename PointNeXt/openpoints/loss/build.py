import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss, L1Loss, HuberLoss
from openpoints.utils import registry
import matplotlib.pyplot as plt
import time

LOSS = registry.Registry('loss')
LOSS.register_module(name='CrossEntropy', module=CrossEntropyLoss)
LOSS.register_module(name='CrossEntropyLoss', module=CrossEntropyLoss)
LOSS.register_module(name='BCEWithLogitsLoss', module=BCEWithLogitsLoss)

#! My implementation
LOSS.register_module(name='MSELoss', module=MSELoss)
LOSS.register_module(name='L1Loss', module=L1Loss)
LOSS.register_module(name='HuberLoss', module=HuberLoss)

@LOSS.register_module()
class ReductionLoss(torch.nn.Module):
    def __init__(self,
                 histogram=None,
                 bins=5,
                 min=0,
                 max=1,
                 reduction=1,
                 _show=False
                 ):
        super(ReductionLoss, self).__init__()
        self.bins = bins
        self.reduction = reduction
        self._show = _show
        """
        ReductionLoss is based on MSE. However, before computing the MSE, predictions
        and targets are converted to classed based on the number of bins. The bins with
        the largest sizes are reduced to equalize the sizes of all bins. The number of
        reduced bins is based in the reduction value.        
        """
        
        # Compute a histogram of reference target values
        self.histogram = histogram
        
        if self._show:
            # Plotting the histogram
            plt.bar(torch.arange(bins), self.histogram)
            plt.xlabel('Bins')
            plt.ylabel('Frequency')
            plt.title('Histogram of Random Data')
            plt.show()
        
        # Sort the histogram based on the sizes
        sorted_bins = torch.argsort(self.histogram)
        
        # Extract the bins that should be reduced
        self.reduction_bins = sorted_bins[-reduction:].cuda()
        
        print(f'Reduction criterion will reduce bin class(es) {self.reduction_bins.tolist()} before computing MSE.')
        
        self.bin_edges = torch.linspace(min, max, steps=bins+1)
    
    def forward(self, pred, target, bins=None):
        if len(pred.shape) > 1:
            pred = pred.view(-1)
            target = target.view(-1)
            
        if bins == None:
            # Index calculation target
            diff = target.unsqueeze(1) - self.bin_edges.unsqueeze(0)
            cumsum = torch.cumsum(diff >= 0, dim=1)
            target_idxs = torch.argmax(cumsum, dim=1)
        else:
            # ? Is it okay to squeeze the batches? This way random points will be picked from all samples before computing mse
            
            
            target_idxs = bins.view(-1)
        
        sample_hist = torch.histc(target, bins=self.bins).long()
        
        # Sort the histogram based on the sizes
        sorted_bins = torch.argsort(sample_hist)
        
        # Compute the maximum bin size
        max_bin = sorted_bins[-self.reduction - 1]
        max_bin_size = self.histogram[max_bin]
        
        # Extract the bins' predictions and classes that should not be reduced
        mask = ~torch.isin(target_idxs, self.reduction_bins)
        temp_pred = pred[mask]
        temp_target = target[mask]
        
        # Iterate over the bins that will be reduced
        for bin in self.reduction_bins:
            # Get the indices of values that belong to this target bin
            bin_mask = target_idxs == bin
            bin_predictions = pred[bin_mask]
            bin_targets = target[bin_mask]
            
            # Randomly generate indices based on the maximum bin size            
            indices = torch.randperm(len(bin_predictions))[:max_bin_size]
            
            # Extract the bin predictions and targets based on the indices
            bin_predictions = bin_predictions[indices]
            bin_targets = bin_targets[indices]
            
            # Concatenate with the values from bins that were not reduced
            temp_pred = torch.cat((temp_pred, bin_predictions))
            temp_target = torch.cat((temp_target, bin_targets))
        
        # Compute the mse
        mse = F.mse_loss(temp_target, temp_pred)
        
        return mse

@LOSS.register_module()     
class DeltaLoss(torch.nn.Module):
    def __init__(self,
                 delta=0.6,
                 power=2,
                 ):
        super(DeltaLoss, self).__init__()
        self.delta = delta
        self.power = power
        """
        DeltaLoss computes the l1 loss for all point for which the target value
        is higher or equal to a delta value. For points with target values lower
        than delta, the mse is computed. Finally the losses are averaged.
        """

    def forward(self, pred, target):
        l1_loss = torch.abs(target - pred) * (target >= self.delta).float()
        
        mse_loss = ((target - pred) ** self.power) * (target < self.delta).float()
        
        return torch.cat((l1_loss, mse_loss)).mean()
        
@LOSS.register_module()
class WeightedMSE(torch.nn.Module):
    def __init__(self, 
                 bins=5,
                 min=0,
                 max=1,
                 weights=[1,1,1,1,1],
                 mode='mean'
                 ):
        super(WeightedMSE, self).__init__()
        self.bins = 5
        self.bin_edges = torch.linspace(min, max, steps=bins+1).cuda()
        self.classes = torch.tensor(range(0, bins)).cuda()
        self.weights = torch.tensor(weights).cuda()
        self.mode = mode
        """
        Compute the weighted MSE loss.
        """
    
    def forward(self, pred, target, bins=None):
        if len(pred.shape) > 1:
            pred = pred.view(-1)
            target = target.view(-1)
    
        if bins == None:
            # Index calculation target
            diff = target.unsqueeze(1) - self.bin_edges.unsqueeze(0)
            cumsum = torch.cumsum(diff >= 0, dim=1)
            target_idxs = torch.argmax(cumsum, dim=1)
        else:
            target_idxs = bins.view(-1)
        
        # Get the weight for each target value
        weights = self.weights[target_idxs]
        
        if self.mode == 'sum':
            return (weights * (pred - target) ** 2).sum() / weights.sum()
        elif self.mode == 'mean':
            return (weights * (pred - target) ** 2).mean()        

@LOSS.register_module()
class SmoothCrossEntropy(torch.nn.Module):
    def __init__(self, label_smoothing=0.2, 
                 ignore_index=None, 
                 num_classes=None, 
                 weight=None, 
                 return_valid=False
                 ):
        super(SmoothCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.return_valid = return_valid
        # Reduce label values in the range of logit shape
        if ignore_index is not None:
            reducing_list = torch.range(0, num_classes).long().cuda(non_blocking=True)
            inserted_value = torch.zeros((1, )).long().cuda(non_blocking=True)
            self.reducing_list = torch.cat([
                reducing_list[:ignore_index], inserted_value,
                reducing_list[ignore_index:]
            ], 0)
        if weight is not None:
            self.weight = torch.from_numpy(weight).float().cuda(
                non_blocking=True).squeeze()
        else:
            self.weight = None
            
    def forward(self, pred, gt):
        if len(pred.shape)>2:
            pred = pred.transpose(1, 2).reshape(-1, pred.shape[1])
        gt = gt.contiguous().view(-1)
        
        if self.ignore_index is not None: 
            valid_idx = gt != self.ignore_index
            pred = pred[valid_idx, :]
            gt = gt[valid_idx]        
            gt = torch.gather(self.reducing_list, 0, gt)
            
        if self.label_smoothing > 0:
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.label_smoothing) + (1 - one_hot) * self.label_smoothing / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            if self.weight is not None:
                loss = -(one_hot * log_prb * self.weight).sum(dim=1).mean()
            else:
                loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gt, weight=self.weight)
        
        if self.return_valid:
            return loss, pred, gt
        else:
            return loss


@LOSS.register_module()
class MaskedCrossEntropy(torch.nn.Module):
    def __init__(self, label_smoothing=0.2):
        super(MaskedCrossEntropy, self).__init__()
        self.creterion = CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def forward(self, logit, target, mask):
        logit = logit.transpose(1, 2).reshape(-1, logit.shape[1])
        target = target.flatten()
        mask = mask.flatten()
        idx = mask == 1
        loss = self.creterion(logit[idx], target[idx])
        return loss

@LOSS.register_module()
class BCELogits(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = BCEWithLogitsLoss(**kwargs)
        
    def forward(self, logits, targets):
        if len(logits.shape)>2:
            logits = logits.transpose(1, 2).reshape(-1, logits.shape[1])
        targets = targets.contiguous().view(-1)
        num_clsses = logits.shape[-1]
        targets_onehot = F.one_hot(targets, num_classes=num_clsses).to(device=logits.device,dtype=logits.dtype)
        return self.criterion(logits, targets_onehot)

@LOSS.register_module()
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, logit, target):
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)  # N,C,H,W => N,C,H*W
            logit = logit.transpose(1, 2)  # N,C,H*W => N,H*W,C
            logit = logit.contiguous().view(-1, logit.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(logit)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != logit.data.type():
                self.alpha = self.alpha.type_as(logit.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()




@LOSS.register_module()
class Poly1CrossEntropyLoss(torch.nn.Module):
    def __init__(self,
                 num_classes: int =50,
                 epsilon: float = 1.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        if len(logits.shape)>2:
            logits = logits.transpose(1, 2).reshape(-1, logits.shape[1])
        labels = labels.contiguous().view(-1)

        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device,
                                                                           dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits,
                             target=labels,
                             reduction='none',
                             weight=self.weight)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1


@LOSS.register_module()
class Poly1FocalLoss(torch.nn.Module):
    def __init__(self,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None,
                 label_is_onehot: bool = False, 
                 **kwargs
                 ):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon. the main one to finetune. larger values -> better performace in imagenet
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        num_classes = logits.shape[1]
        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                labels = F.one_hot(labels.unsqueeze(1), num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device, dtype=logits.dtype)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1

@LOSS.register_module()
class MultiShapeCrossEntropy(torch.nn.Module):
    def __init__(self, criterion_args, **kwargs):
        super(MultiShapeCrossEntropy, self).__init__()
        self.criterion = build_criterion_from_cfg(criterion_args)

    def forward(self, logits_all_shapes, points_labels, shape_labels):
        batch_size = shape_labels.shape[0]
        losses = 0
        for i in range(batch_size):
            sl = shape_labels[i]
            logits = torch.unsqueeze(logits_all_shapes[sl][i], 0)
            pl = torch.unsqueeze(points_labels[i], 0)
            loss = self.criterion(logits, pl)
            losses += loss
        return losses / batch_size

def build_criterion_from_cfg(cfg, **kwargs):
    """
    Build a criterion (loss function), defined by cfg.NAME.
    Args:
        cfg (eDICT): 
    Returns:
        criterion: a constructed loss function specified by cfg.NAME
    """
    
    return LOSS.build(cfg, **kwargs)