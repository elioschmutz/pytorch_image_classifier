from collections import OrderedDict
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class ModelBuilder():
    SUPPORTED_MODEL_TYPES = ['vgg11', 'vgg16']
    SUPPORTED_OPTIMIZER_TYPES = ['sgd', 'adam']
    SUPPORTED_CRITERION_TYPES = ['nllloss']

    def __init__(self):
        self.model = None
        self.builder_params = {}

    def set_builder_params(self, params):
        self.builder_params = params
        return self

    def with_model(self, model_type, out_features, dropout=0.5, hidden_layers=[]):
        if model_type not in self.SUPPORTED_MODEL_TYPES:
            raise Exception(f"Model type is not supported. Supported types are: {self.SUPPORTED_MODEL_TYPES}")

        self.builder_params['model_type'] = model_type
        self.builder_params['out_features'] = out_features
        self.builder_params['dropout'] = dropout
        self.builder_params['hidden_layers'] = hidden_layers

        return self

    def with_optimizer(self, optimizer_type, **kwargs):
        if optimizer_type not in self.SUPPORTED_OPTIMIZER_TYPES:
            raise Exception(
                f"Optimizer type is not supported. Supported types are: {self.SUPPORTED_OPTIMIZER_TYPES}")

        self.builder_params['optimizer_type'] = optimizer_type
        self.builder_params['optimizer_params'] = kwargs

        return self

    def with_criterion(self, criterion_type, **kwargs):
        if criterion_type not in self.SUPPORTED_CRITERION_TYPES:
            raise Exception(
                f"Criterion type is not supported. Supported types are: {self.SUPPORTED_CRITERION_TYPES}")

        self.builder_params['criterion_type'] = criterion_type
        self.builder_params['criterion_params'] = kwargs

        return self

    def get_model_train_params(self):
        return self.model.classifier.parameters()

    def build_model(self):
        model_type = self.builder_params['model_type']
        out_features = self.builder_params['out_features']
        dropout = self.builder_params['dropout']
        hidden_layers = self.builder_params['hidden_layers']

        if model_type == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif model_type == 'vgg11':
            model = models.vgg11(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        next_in_features = model.classifier[0].in_features
        next_out_features = hidden_layers[0] if hidden_layers else out_features

        layers = OrderedDict([])
        layers_list = [*hidden_layers, out_features]
        next_in_features = model.classifier[0].in_features

        for i, layer in enumerate(layers_list):
            next_out_features = layer
            layers.update([
                (f'l_linear_{i}', nn.Linear(next_in_features, next_out_features))
            ])

            if i + 1 < len(layers_list):
                layers.update([
                    (f'l_relu_{i}', nn.ReLU()),
                    (f'l_dropout_{i}', nn.Dropout(dropout)),
                ])

            next_in_features = layer

        layers.update([
            ('errorfunction', nn.LogSoftmax(dim=1)),
        ])

        model.classifier = nn.Sequential(layers)
        self.model = model

        return model

    def build_optimizier(self):
        optimizer_type = self.builder_params['optimizer_type']
        optimizer_params = self.builder_params['optimizer_params']

        if optimizer_type == 'adam':
            optimizer = optim.Adam(self.get_model_train_params(), **optimizer_params)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(self.get_model_train_params(), **optimizer_params)

        return optimizer

    def build_criterion(self):
        criterion_type = self.builder_params['criterion_type']
        criterion_params = self.builder_params['criterion_params']

        if criterion_type == 'nllloss':
            criterion = nn.NLLLoss()

        return criterion

    def build(self):
        return self.build_model(), self.build_optimizier(), self.build_criterion()

    def copy(self):
        return ModelBuilder().set_builder_params(self.builder_params)
