import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model, extracted_layers):
        super(ResNetFeatureExtractor, self).__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = model.fc
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        if 'conv1' in self.extracted_layers:
            outputs += [x]
            if len(outputs) == len(self.extracted_layers):
                return outputs
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if 'maxpool' in self.extracted_layers:
            outputs += [x]
            if len(outputs) == len(self.extracted_layers):
                return outputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if 'layer3' in self.extracted_layers:
            outputs += [x]
            if len(outputs) == len(self.extracted_layers):
                return outputs
        x = self.layer4(x)
        if 'layer4' in self.extracted_layers:
            outputs += [x]
            if len(outputs) == len(self.extracted_layers):
                return outputs

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if 'avgpool' in self.extracted_layers:
            outputs += [x]
        if 'fc' in self.extracted_layers:
            x = self.fc(x)
            outputs += [x]
        return outputs

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, model, extracted_layers, transform_input):
        super(InceptionFeatureExtractor, self).__init__()
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c
        self.fc = model.fc
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        if 'mix7' in self.extracted_layers:
            outputs += [x]
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        if 'avgpool' in self.extracted_layers:
            outputs += [x]
        # 2048
        if 'fc' in self.extracted_layers:
            x = self.fc(x)
            outputs += [x]
        # 1000 (num_classes)
        return outputs

class DensenetFeatureExtractor(nn.Module):
    def __init__(self, model, extracted_layers):
        super(DensenetFeatureExtractor, self).__init__()
        self.extracted_layers = extracted_layers
        self.features = model.features
        self.classifier = model.classifier

    def forward(self, x):
        outputs = []
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        if 'avgpool' in self.extracted_layers:
            outputs += [out]
        if 'fc' in self.extracted_layers:
            out = self.classifier(out)
            outputs += [out]
        return outputs