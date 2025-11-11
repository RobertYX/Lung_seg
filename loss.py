import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.nn.functional import one_hot
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss


SUPPORTED_WEIGHTING = ['default', 'GDL']

class FocalLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', gamma=1.5, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        input = torch.sigmoid(input)
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
            pred.double().sum() + truth.double().sum() + 1
        )

        return bce_loss + (1 - dice_coef)
    
class WCEDCELoss(nn.Module):
    def __init__(self, num_classes=1, inter_weights=0.5, intra_weights=None, device='cuda', cf='ce'):
        super(WCEDCELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=intra_weights)
        self.fc_loss = FocalLoss(weight=intra_weights)
        self.num_classes = num_classes
        self.intra_weights = intra_weights
        self.inter_weights = inter_weights
        self.device = device
        self.cf = cf

    def dice_loss(self, prediction, target, weights):
        """Calculating the dice loss
        Args:
            prediction = predicted image
            target = Targeted image
        Output:
            dice_loss"""
        smooth = 1e-5
        prediction = torch.softmax(prediction, dim=1)
        batchsize = target.size(0)
        num_classes = target.size(1)
        prediction = prediction.view(batchsize, num_classes, -1)
        target = target.view(batchsize, num_classes, -1)

        intersection = (prediction * target)

        dice = (2. * intersection.sum(2) + smooth) / (prediction.sum(2) + target.sum(2) + smooth)
        #print('dice: ', dice)
        dice_loss = 1 - dice.sum(0) / batchsize
        weighted_dice_loss = dice_loss * weights

        # print(dice_loss, weighted_dice_loss)
        return weighted_dice_loss.mean()

    def forward(self, pred, label):
        """Calculating the loss and metrics
            Args:
                prediction = predicted image
                target = Targeted image
                metrics = Metrics printed
                bce_weight = 0.5 (default)
            Output:
                loss : dice loss of the epoch """
        if self.cf == "ce":
            cel = self.ce_loss(pred, label)
        elif self.cf == "fc" :
            cel = self.fc_loss(pred, label)
        else:
            cel = None
            print("None loss")
            exit(0)
        # temp = label.cpu().data.numpy()
        # a = len(temp[temp==0]); b = len(temp[temp==1])
        # ratio = torch.tensor([a/(a+b), b/(a+b)]).to(self.device)
        if len(label.shape)==3:
            label_onehot = one_hot(label, num_classes=self.num_classes).permute(0, 3, 1, 2).contiguous()
        if len(label.shape)==4:
            label_onehot = one_hot(label, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).contiguous()
        dicel = self.dice_loss(pred, label_onehot, self.intra_weights)#ratio
        #print('ce: ', cel*self.inter_weights, 'dicel: ', dicel * (1 - self.inter_weights))
        loss = cel * self.inter_weights + dicel * (1 - self.inter_weights)
        return loss
    
class MSELoss(nn.Module):
    def __init__(self, num_classes=2, intra_weights=None, device='cuda'):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(weight=intra_weights)
        self.num_classes = num_classes

        self.device = device
     
    def forward(self, pred, label):
        # print(label.shape)
        label_onehot = one_hot(label, num_classes=self.num_classes).permute(0, 3, 1, 2).contiguous()
        loss = self.mse_loss(pred, label_onehot.float())
        return loss

class GeneralizedWassersteinDiceLoss(_Loss):
    """
    Generalized Wasserstein Dice Loss [1] in PyTorch.
    Optionally, one can use a weighting method for the
    class-specific sum of errors similar to the one used
    in the generalized Dice Loss [2].
    For this behaviour, please use weighting_mode='GDL'.
    The exact formula of the Wasserstein Dice loss in this case
    can be found in the Appendix of [3].
    References:
    ===========
    [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks."
        Fidon L. et al. MICCAI BrainLes (2017).
    [2] "Generalised dice overlap as a deep learning loss function
        for highly unbalanced segmentations."
        Sudre C., et al. MICCAI DLMIA (2017).
    [3] "Comparative study of deep learning methods for the automatic
        segmentation of lung, lesion and lesion type in CT scans of
        COVID-19 patients."
        Tilborghs, S. et al. arXiv preprint arXiv:2007.15546 (2020).
    """
    def __init__(self, dist_matrix, weighting_mode='default', reduction='mean'):
        """
        :param dist_matrix: 2d tensor or 2d numpy array; matrix of distances
        between the classes.
        It must have dimension C x C where C is the number of classes.
        :param: weighting_mode: str; indicates how to weight the class-specific
        sum of errors.
        'default' corresponds to the GWDL used in the original paper [1],
        'GDL' corresponds to the GWDL used in [2].
        :param reduction: str; reduction mode.
        References:
        ===========
        [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
            Segmentation using Holistic Convolutional Networks."
            Fidon L. et al. MICCAI BrainLes (2017).
        [2] "Comparative study of deep learning methods for the automatic
            segmentation of lung, lesion and lesion type in CT scans of
            COVID-19 patients."
            Tilborghs, S. et al. arXiv preprint arXiv:2007.15546 (2020).
        """
        super(GeneralizedWassersteinDiceLoss, self).__init__(
            reduction=reduction)

        assert weighting_mode in SUPPORTED_WEIGHTING, \
            "weighting_mode must be in %s" % str(SUPPORTED_WEIGHTING)

        self.M = dist_matrix
        if isinstance(self.M, np.ndarray):
            self.M = torch.from_numpy(self.M)
        if torch.max(self.M) != 1:
            print('Normalize the maximum of the distance matrix '
                  'used in the Generalized Wasserstein Dice Loss to 1.')
            self.M = self.M / torch.max(self.M)

        self.num_classes = self.M.size(0)
        self.alpha_mode = weighting_mode
        self.reduction = reduction

    def forward(self, input, target):
        """
        Compute the Generalized Wasserstein Dice loss
        between input and target tensors.
        :param input: tensor. input is the scores maps (before softmax).
        The expected shape of input is (N, C, H, W, D) in 3d
        and (N, C, H, W) in 2d.
        :param target: target is the target segmentation.
        The expected shape of target is (N, H, W, D) or (N, 1, H, W, D) in 3d
        and (N, H, W) or (N, 1, H, W) in 2d.
        :return: scalar tensor. Loss function value.
        """
        epsilon = np.spacing(1)  # smallest number available

        # Convert the target segmentation to long if needed
        target = target.long()

        # Aggregate spatial dimensions
        flat_input = input.view(input.size(0), input.size(1), -1)  # b,c,s
        flat_target = target.view(target.size(0), -1)  # b,s

        # Apply the softmax to the input scores map
        probs = F.softmax(flat_input, dim=1)  # b,c,s

        # Compute the Wasserstein distance map
        wass_dist_map = self.wasserstein_distance_map(probs, flat_target)

        # Compute the generalised number of true positives
        alpha = self.compute_alpha_generalized_true_positives(flat_target)

        # Compute the Generalized Wasserstein Dice loss
        if self.alpha_mode == 'GDL':
            # use GDL-style alpha weights (i.e. normalize by the volume of each class)
            # contrary to [1] we also use alpha in the "generalized all error".
            true_pos = self.compute_generalized_true_positive(
                alpha, flat_target, wass_dist_map)
            denom = self.compute_denominator_GDL(alpha, flat_target, wass_dist_map)
        else:  # default: as in [1]
            # (i.e. alpha=1 for all foreground classes and 0 for the background).
            # Compute the generalised number of true positives
            true_pos = self.compute_generalized_true_positive(
                alpha, flat_target, wass_dist_map)
            all_error = torch.sum(wass_dist_map, dim=1)
            denom = 2 * true_pos + all_error
        wass_dice = (2. * true_pos + epsilon) / (denom + epsilon)
        wass_dice_loss = 1. - wass_dice

        if self.reduction == 'sum':
            return wass_dice_loss.sum()
        elif self.reduction == 'none':
            return wass_dice_loss
        else:  # default is mean reduction
            return wass_dice_loss.mean()

    def wasserstein_distance_map(self, flat_proba, flat_target):
        """
        Compute the voxel-wise Wasserstein distance (eq. 6 in [1]) for
        the flattened prediction and the flattened labels (ground_truth)
        with respect to the distance matrix on the label space M.
        References:
        ===========
        [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks",
        Fidon L. et al. MICCAI BrainLes 2017
        """
        # Turn the distance matrix to a map of identical matrix
        M_extended = torch.clone(self.M).to(flat_proba.device)
        M_extended = torch.unsqueeze(M_extended, dim=0)  # C,C -> 1,C,C
        M_extended = torch.unsqueeze(M_extended, dim=3)  # 1,C,C -> 1,C,C,1
        M_extended = M_extended.expand((
            flat_proba.size(0),
            M_extended.size(1),
            M_extended.size(2),
            flat_proba.size(2)
        ))

        # Expand the feature dimensions of the target
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        flat_target_extended = flat_target_extended.expand(  # b,1,s -> b,C,s
            (flat_target.size(0), M_extended.size(1), flat_target.size(1))
        )
        flat_target_extended = torch.unsqueeze(flat_target_extended, dim=1)  # b,C,s -> b,1,C,s

        # Extract the vector of class distances for the ground-truth label at each voxel
        M_extended = torch.gather(M_extended, dim=1, index=flat_target_extended)  # b,C,C,s -> b,1,C,s
        M_extended = torch.squeeze(M_extended, dim=1)  # b,1,C,s -> b,C,s

        # Compute the wasserstein distance map
        wasserstein_map = M_extended * flat_proba

        # Sum over the classes
        wasserstein_map = torch.sum(wasserstein_map, dim=1)  # b,C,s -> b,s
        return wasserstein_map

    def compute_generalized_true_positive(self, alpha, flat_target, wasserstein_distance_map):
        # Extend alpha to a map and select value at each voxel according to flat_target
        alpha_extended = torch.unsqueeze(alpha, dim=2)  # b,C -> b,C,1
        alpha_extended = alpha_extended.expand(  # b,C,1 -> b,C,s
            (flat_target.size(0), self.num_classes, flat_target.size(1))
        )
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        alpha_extended = torch.gather(
            alpha_extended, index=flat_target_extended, dim=1)  # b,C,s -> b,1,s
        alpha_extended = torch.squeeze(alpha_extended, dim=1) # b,1,s -> b,s

        # Compute the generalized true positive as in eq. 9 of [1]
        # The value in the parenthesis is 1 because the distance from a foreground class
        # to the background is always equal to 1 (see __init__ function)
        # and because the value of alpha for the background class is 0
        generalized_true_pos = torch.sum(
            alpha_extended * (1. - wasserstein_distance_map),
            dim=1,
        )

        return generalized_true_pos

    def compute_denominator_GDL(self, alpha, flat_target, wasserstein_distance_map):
        # Extend alpha to a map and select value at each voxel according to flat_target
        alpha_extended = torch.unsqueeze(alpha, dim=2)  # b,C -> b,C,1
        alpha_extended = alpha_extended.expand(  # b,C,1 -> b,C,s
            (flat_target.size(0), self.num_classes, flat_target.size(1))
        )
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        alpha_extended = torch.gather(
            alpha_extended, index=flat_target_extended, dim=1)  # b,C,s -> b,1,s
        alpha_extended = torch.squeeze(alpha_extended, dim=1) # b,1,s -> b,s

        generalized_true_pos = torch.sum(
            alpha_extended * (2. - wasserstein_distance_map),
            dim=1,
        )

        return generalized_true_pos

    def compute_alpha_generalized_true_positives(self, flat_target):
        """
        Compute the weights alpha_l of eq. 9 in [1].
        References:
        ===========
        [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks",
        Fidon L. et al. MICCAI BrainLes 2017.
        """
        if self.alpha_mode == 'GDL':  # GDL style
            # Define alpha like in the generalized dice loss
            # i.e. the inverse of the volume of each class.
            # Convert target to one-hot class encoding.
            one_hot = F.one_hot(  # shape: b,c,s
                flat_target, num_classes=self.num_classes).permute(0, 2, 1).float()
            volumes = torch.sum(one_hot, dim=2)  # b,c
            alpha = 1. / (volumes + 1.)
        else:  # default, i.e. as in [1]
            # alpha weights are 0 for the background and 1 otherwise
            alpha_np = np.ones((flat_target.size(0), self.num_classes))  # b,c
            alpha_np[:, 0] = 0.
            alpha = torch.from_numpy(alpha_np).float()
            if torch.cuda.is_available():
                alpha = alpha.cuda()
        return alpha    
    
if __name__ == '__main__':

    dist_mat = np.array([
        [0., 1., 1.],
        [1., 0., 0.5],
        [1., 0.5, 0.]
    ])
    wass_loss = GeneralizedWassersteinDiceLoss(dist_matrix=dist_mat)
    # 1D prediction; shape: batch size, n class, n elements
    pred = torch.tensor([[[1, 0], [0, 1], [0, 0]]], dtype=torch.float32).cuda()
    # !D ground truth; shape: batch size, n elements 
    grnd = torch.tensor([[0, 2]], dtype=torch.int64).cuda()

    wass_loss = GeneralizedWassersteinDiceLoss(dist_matrix=dist_mat)

    loss = wass_loss(pred, grnd)
    print(pred.shape, grnd.shape)

    print(loss.item())

