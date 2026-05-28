import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def _expand_image_stats(image_stats, in_channels):
    if image_stats is None:
        return None
    if len(image_stats) == in_channels:
        return list(image_stats)
    if len(image_stats) > in_channels:
        return list(image_stats[:in_channels])
    extra = [image_stats[-1]] * (in_channels - len(image_stats))
    return list(image_stats) + extra

def _adapt_backbone_input_channels(model, in_channels, pretrained):
    if in_channels == 3:
        return

    conv1 = model.backbone.body.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=False,
    )

    with torch.no_grad():
        if pretrained:
            # Reuse RGB pretrained filters and initialize extra channels by RGB mean.
            copy_channels = min(3, in_channels)
            new_conv.weight[:, :copy_channels] = conv1.weight[:, :copy_channels]
            if in_channels > 3:
                mean_w = conv1.weight.mean(dim=1, keepdim=True)
                for ch in range(3, in_channels):
                    new_conv.weight[:, ch : ch + 1] = mean_w
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

    model.backbone.body.conv1 = new_conv

def get_model(
    num_classes,
    pretrained=True,
    in_channels=3,
    image_mean=None,
    image_std=None,
):
    """
    Create Mask R-CNN model with ResNet-50 backbone

    Args:
        num_classes (int): Number of classes (including background)
        pretrained (bool): Whether to use pretrained weights
        in_channels (int): Number of input channels
        image_mean (list[float] | None): Input normalization mean
        image_std (list[float] | None): Input normalization std

    Returns:
        model: Mask R-CNN model
    """
    # Load pre-trained Mask R-CNN model (new torchvision API)
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights, weights_backbone=None)
    _adapt_backbone_input_channels(model, in_channels=in_channels, pretrained=pretrained)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    # Ensure model transform stats match the number of input channels
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
    model.transform.image_mean = _expand_image_stats(image_mean or default_mean, in_channels)
    model.transform.image_std = _expand_image_stats(image_std or default_std, in_channels)

    return model

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """Create warmup learning rate scheduler"""
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return optim.lr_scheduler.LambdaLR(optimizer, f)

def load_model(model_path, device, num_classes, pretrained=True, in_channels=3):
    model = get_model(
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
