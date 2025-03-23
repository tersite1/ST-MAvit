import os
import logging
import torch
import torch.nn as nn
from metrics.registry import DETECTOR
from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name="sew_resnet")
class SewResnetDetector(AbstractDetector):  # 클래스 이름 통일
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        return backbone

    def build_loss(self, config):
        loss_class = LOSSFUNC[config["loss_func"]]
        loss_func = loss_class(config['label_smoothing'])

        return loss_func

    def features(self, data_dict: dict):
        return self.backbone.features(data_dict['image'])

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.classifier(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        return {'overall': loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        probs = torch.softmax(pred, dim=1)  # shape: (B, 2)
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), probs.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {'cls': pred, 'prob': prob, 'feat': features}


# 테스트 코드
if __name__ == "__main__":
    from loss.ce_loss import CELoss

    # 임시 config
    config = {
        "backbone_name": "sew_resnet",
        "backbone_config": {
            "num_classes": 2,
            "model_size": "sew_resnet18"
        },
        "loss_func": "ce_loss",  # 'ce_loss'가 LOSSFUNC에 등록되어 있어야 함
        "label_smoothing": 0.01
    }

    model = SewResnetDetector(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Dummy input
    dummy_input = {
        "image": torch.randn(4, 3, 256, 256).to(device),  # batch of 4
        "label": torch.randint(0, 2, (4,)).to(device)
    }

    with torch.no_grad():
        pred_dict = model(dummy_input)
        loss = model.get_losses(dummy_input, pred_dict)
        metrics = model.get_train_metrics(dummy_input, pred_dict)

    print("Predictions:", pred_dict['cls'].shape)
    print("Probabilities:", pred_dict['prob'].shape)
    print("Loss:", loss)
    print("Metrics:", metrics)
