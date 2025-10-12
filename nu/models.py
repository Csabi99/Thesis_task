from flwr.common.logger import log
from logging import INFO, DEBUG, ERROR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
from torchvision.models import MobileNet_V2_Weights

class LightweightClassifier(nn.Module):
    def __init__(self, dev, checkpoint=False, input_shape=(3, 80, 45), output_dim=4):
        super(LightweightClassifier, self).__init__()
        if checkpoint:
            log(INFO, f"LightweightClassifier should be used with checkpoint, but it is not implemented yet.    ")
        # Convs
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        
        # figure out flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)  # batch=1, shape=(3,80,45)
            dummy_out = self._forward_convs(dummy)
            flatten_dim = dummy_out.view(1, -1).size(1)
        
        # FC layers
        self.fc1 = nn.Linear(flatten_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

        self.dropout = nn.Dropout(0.5)
        self._dev = dev
        self._global_model = None
        self._run_mode = None

    def _forward_convs(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        return x

    def forward(self, x):
        x = self._forward_convs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def set_global_model(self, model):
        """Set the global model for the client."""
        self._global_model = model

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.state_dict().items()]        


class TransferClassifier(nn.Module):
    def __init__(self, dev, checkpoint=False, input_shape=(3, 80, 45), num_classes=4, freeze_backbone=True, yolo=False):
        super(TransferClassifier, self).__init__()
        self._dev = dev
        if checkpoint:
            log(INFO, f"TransferClassifier with checkpoint")
            model_path = "/app/models/TransferClassifier.pt"
            state_dict = torch.load(model_path, map_location="cpu")
            self.load_state_dict(state_dict)
        else:
            log(INFO, f"TransferClassifier from scratch")
            #backbone_weights = torch.load("/app/models/mobilenet_v2-b0353104.pth", map_location="cpu")
            #self.backbone.load_state_dict(backbone_weights)

            self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            if freeze_backbone:
                for param in self.backbone.features.parameters():
                    param.requires_grad = False
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes)
            )            
        self.to(self._dev)
        self._global_model = None
        self._run_mode = None

    def forward(self, x):
        return self.backbone(x)

    def set_global_model(self, model):
        self._global_model = model

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.backbone.classifier.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.backbone.classifier.state_dict()
        new_state_dict = {k: torch.tensor(v) for k, v in zip(state_dict.keys(), parameters)}
        new_state_dict = {k: v.to(self._dev) for k, v in new_state_dict.items()}
        self.backbone.classifier.load_state_dict(new_state_dict)        


class TransferYoloClassifier(nn.Module):
    def __init__(self, dev, checkpoint=False, input_shape=(3, 80, 80), num_classes=4, freeze_backbone=True):
        super(TransferYoloClassifier, self).__init__()
        self._dev = dev

        if checkpoint:
            log(INFO, f"TransferYoloClassifier with checkpoint")
            model_path = "/app/models/TransferYoloClassifier.pt"
            state_dict = torch.load(model_path, map_location="cpu")
            self.load_state_dict(state_dict)
        else:
            log(INFO, f"TransferYoloClassifier from scratch")

            # --- Load YOLOv5 backbone using torch.hub ---
            self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            log(INFO, f"Loaded YOLOv5 backbone")

            # --- Keep only the feature extractor (exclude YOLO detection head) ---
            self.backbone = nn.Sequential(*list(self.yolo.model.children())[:-1])

            # --- Optionally freeze the backbone ---
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                log(INFO, "YOLO backbone frozen")

            # --- Determine flattened feature size dynamically ---
            dummy = torch.randn(1, *input_shape)
            with torch.no_grad():
                features = self.backbone(dummy)
                if isinstance(features, (list, tuple)):
                    features = features[-1]
                feature_dim = features.shape[1] * features.shape[2] * features.shape[3]

            log(INFO, f"YOLO feature dimension: {feature_dim}")

            # --- Define classifier head ---
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes)
            )

        self.to(self._dev)
        self._global_model = None
        self._run_mode = None

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
        x = self.classifier(x)
        return x

    def set_global_model(self, model):
        self._global_model = model

    def get_parameters(self):
        # Return only classifier weights for federated learning
        return [val.cpu().numpy() for _, val in self.classifier.state_dict().items()]

    def set_parameters(self, parameters):
        # Update classifier weights only
        state_dict = self.classifier.state_dict()
        new_state_dict = {k: torch.tensor(v) for k, v in zip(state_dict.keys(), parameters)}
        new_state_dict = {k: v.to(self._dev) for k, v in new_state_dict.items()}
        self.classifier.load_state_dict(new_state_dict)
