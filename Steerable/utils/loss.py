import torch
import torch.nn as nn
import torch.nn.functional as F


#####################################################################################################
######################################### Focal Loss ################################################
##################################################################################################### 
    
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha if torch.is_tensor(alpha) else torch.tensor(alpha)
        self.gamma = gamma

    def forward(self, preds, truth):
        ce_loss = torch.nn.functional.cross_entropy(preds, truth, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha.ndimension() == 0:
            focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        else:
            assert len(self.alpha) == preds.shape[1]
            self.alpha = self.alpha.to(truth.device)
            alpha_t = self.alpha[truth]
            focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()
    
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, truth):
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(preds.squeeze(1), truth.float(), reduction='none')
        focal_loss = self.alpha * ((1 - torch.exp(-ce_loss)) ** self.gamma) * ce_loss
        return focal_loss.mean()

#####################################################################################################
########################################## Dice Loss ################################################
##################################################################################################### 

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = F.softmax(preds, dim=1)
        preds = preds.flatten(2)
        
        targets_one_hot = F.one_hot(targets, num_classes=preds.size(1)).permute(0, 3, 1, 2).float()
        targets = targets_one_hot.flatten(2)

        # Compute Dice Loss
        intersection = (preds * targets).sum(dim=2)
        union = preds.sum(dim=2) + targets.sum(dim=2)
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coeff.mean(dim=1)

        return dice_loss.mean()
    

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds_flat = torch.sigmoid(preds).flatten(1)
        targets_flat = targets.flatten(1)
        intersection = (preds_flat * targets_flat).sum(dim=-1)
        dice_coeff = (2. * intersection + self.smooth) / (preds_flat.sum(dim=-1) + targets_flat.sum(dim=-1) + self.smooth)
        
        return 1 - dice_coeff.mean()
    
#####################################################################################################
####################################### Jaccard Loss ################################################
##################################################################################################### 

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = F.softmax(preds, dim=1)
        preds = preds.flatten(2)
        
        targets_one_hot = F.one_hot(targets, num_classes=preds.size(1)).permute(0, 3, 1, 2).float()
        targets = targets_one_hot.flatten(2)

        # Compute Dice Loss
        intersection = (preds * targets).sum(dim=2)
        union = preds.sum(dim=2) + targets.sum(dim=2)
        jaccard_coeff = (intersection + self.smooth) / (union - intersection + self.smooth)
        jaccard_loss = 1 - jaccard_coeff.mean(dim=1)

        return jaccard_loss.mean()
    

class BinaryJaccardLoss(nn.Module):
    def __init__(self, smooth=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds_flat = torch.sigmoid(preds).flatten(1)
        targets_flat = targets.flatten(1)
        intersection = (preds_flat * targets_flat).sum(dim=-1)
        union = preds_flat.sum(dim=-1) + targets_flat.sum(dim=-1)
        jaccard_coeff = (intersection + self.smooth) / (union - intersection + self.smooth)
        
        return 1 - jaccard_coeff.mean()

    
#####################################################################################################
################################## Pixel Cross Entropy Loss #########################################
##################################################################################################### 
    
class PixelCrossEntropyLoss(nn.Module):
    def __init__(self) -> None:
        super(PixelCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, pred, truth):
        loss = self.criterion(pred, truth)
        
        return loss
    
class BinaryPixelCrossEntropyLoss(nn.Module):
    def __init__(self) -> None:
        super(BinaryPixelCrossEntropyLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, truth):
        loss = self.criterion(pred, truth)
        
        return loss
  
#####################################################################################################
########################################## Combined Loss ############################################
#####################################################################################################   
    
class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5, smooth=1):
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dice_loss = DiceLoss()
        self.ce_loss = PixelCrossEntropyLoss()

    def forward(self, preds, truth):
        ce_loss = self.ce_loss(preds, truth)
        dice_loss = self.dice_loss(preds, truth)
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss
    
class BinaryCombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5, smooth=1):
        super(BinaryCombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dice_loss = BinaryDiceLoss()
        self.ce_loss = BinaryPixelCrossEntropyLoss()

    def forward(self, preds, truth):
        ce_loss = self.ce_loss(preds, truth)
        dice_loss = self.dice_loss(preds, truth)
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss
    
#####################################################################################################
######################################### Tversky Loss ##############################################
##################################################################################################### 
    
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = F.softmax(preds, dim=1)
        preds = preds.flatten(2)
        
        targets_one_hot = F.one_hot(targets, num_classes=preds.size(1)).permute(0, 3, 1, 2).float()
        targets = targets_one_hot.flatten(2)

        tp = (preds * targets).sum(dim=2)
        fp = ((1 - targets) * preds).sum(dim=2)
        fn = (targets * (1 - preds)).sum(dim=2)

        # Calculate Tversky loss
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        tversky_loss = 1 - tversky_index.mean(dim=1)

        return tversky_loss.mean()
    
class BinaryTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, preds, targets):
        preds_flat = torch.sigmoid(preds).flatten(1)
        targets_flat = targets.flatten(1)

        tp = (preds * targets).sum(dim=-1)
        fp = ((1 - targets) * preds).sum(dim=-1)
        fn = (targets * (1 - preds)).sum(dim=-1)

        # Calculate Tversky loss
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return tversky_index.mean()

