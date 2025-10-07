import torch

class Metrics:
    def __init__(self, num_classes, metric='dice'):
        if num_classes<=0:
            raise ValueError(f"num_classes {num_classes} must be an integer greated than 0.")
        if metric not in ['dice', 'iou', 'accuracy']:
            raise ValueError(f"metric ({metric}) must be one of 'dice', 'iou' or 'accuracy'.")
        
        self.num_classes = num_classes
        self.metric = metric
        self.confusion = torch.zeros(1, self.num_classes, self.num_classes, dtype=torch.int64)
        
        
    ################## Confusion Matrix
    def get_confusion_matrix(self, preds, targets):
        if preds is None and targets is None:
            return self.confusion
        elif preds is not None and targets is not None:
            if not preds.shape == targets.shape:
                raise ValueError(f"Size of prediction {list(preds.shape)} must match size of targets {list(targets.shape)}")
            
            conf = torch.zeros(targets.shape[0], self.num_classes, self.num_classes, dtype=torch.int64)
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    conf[:,i,j] = torch.logical_and(targets==i, preds==j).reshape(preds.shape[0], -1).sum(dim=-1)
            return conf
        else:
            raise ValueError('Either both preds and targets should be given or none of them.')
        
    def add_to_confusion_matrix(self, preds, targets):
        self.confusion += torch.sum(self.get_confusion_matrix(preds, targets), dim=0, keepdim=True)

    ################## Metrics
    def macro_per_class(self, preds=None, targets=None):
        conf = self.get_confusion_matrix(preds, targets)
        tp = torch.diagonal(conf, dim1=-2, dim2=-1)
        fp = torch.sum(conf, dim=-1) - tp
        fn = torch.sum(conf, dim=-2) - tp
        
        return torch.nan_to_num(self._get_score(tp,fp,fn), 1).mean(dim=0)

    def macro(self, preds=None, targets=None):
        return torch.mean(self.macro_per_class(preds, targets)).item()
    
    def micro_per_class(self, preds=None, targets=None):
        conf = self.get_confusion_matrix(preds, targets)
        tp = torch.diagonal(conf, dim1=-2, dim2=-1).sum(dim=0)
        fp = (torch.sum(conf, dim=-1) - tp).sum(dim=0)
        fn = (torch.sum(conf, dim=-2) - tp).sum(dim=0)

        return torch.nan_to_num(self._get_score(tp,fp,fn), 1)
    
    def micro(self, preds=None, targets=None):
        conf = self.get_confusion_matrix(preds, targets)
        tp = torch.diagonal(conf, dim1=-2, dim2=-1)
        fp = (torch.sum(conf, dim=-1) - tp).sum()
        fn = (torch.sum(conf, dim=-2) - tp).sum()
        tp = tp.sum()
            
        return torch.nan_to_num(self._get_score(tp,fp,fn), 1).item()

    def _get_score(self, tp, fp, fn):
        if self.metric == 'dice':
            score = 2*tp / (2*tp + fp + fn)

        elif self.metric == 'iou':
            score = tp / (tp + fp + fn)
        
        elif self.metric == 'accuracy':
            score = tp / (tp + fn)
            
        return score
