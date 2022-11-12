import torch.nn as nn
from torch.optim import Adam
from torchvision import models
from pytorch_lightning import LightningModule
import torchmetrics as tm

from models.utils import log_metrics

class Efficientnet(LightningModule):
    def __init__(self, num_classes=14, class_names=[], feature_extract=False, master_log=None, **kwargs) -> None:
        super(Efficientnet, self).__init__()
        assert len(class_names) == num_classes, "Number of class names must match number of classes"
        self.class_names = class_names
        self.num_classes = num_classes
        self.loss = nn.CrossEntropyLoss()
        self.model, self.input_size = self.initialize_model(num_classes, feature_extract)
        self.master_log = master_log
        # Test Metrics
        self.test_acc = tm.Accuracy(num_classes=num_classes, average='macro')
        self.test_f1 = tm.F1Score(num_classes=num_classes, average='macro')
        self.test_auc = tm.AUROC(num_classes=num_classes, average='macro')
        self.test_prec = tm.Precision(num_classes=num_classes, average='macro')
        self.test_recall = tm.Recall(num_classes=num_classes, average='macro')

        self.save_hyperparameters(ignore=[
            'feature_extract'
        ])


    def initialize_model(self, num_classes, feature_extract, use_pretrained=True):
        model_ft = models.efficientnet_b0(weights=use_pretrained)
        self.set_parameter_requires_grad(model_ft, feature_extract)
        print(len(model_ft.classifier))
        num_ftrs = model_ft.classifier[-1].in_features
        model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
            
        return model_ft, input_size


    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    def _common_step(self, batch, batch_nb):
        inputs, category_labels, _ = batch

        outputs = self.model(inputs)
        loss = self.loss(outputs, category_labels)

        # return loss, outputs, category_labels
        return {'loss': loss, 'outputs': outputs, 'labels': category_labels}


    def train_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx)


    def train_epoch_end(self, epoch_output):
        stage = 'train'
        res = log_metrics(self, epoch_output, stage)
        return res


    def val_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx)


    def val_epoch_end(self, outputs):
        stage = "val"
        res = log_metrics(self, outputs, stage)
        return res


    def testing_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx)


    def testing_end(self, outputs):
        stage = "test"
        res = log_metrics(self, outputs, stage)
        return res


    def optimizer(self, parameters, lr, weight_decay):
        return Adam(parameters, lr=lr, weight_decay=weight_decay)



if __name__ == '__main__':
    # Initialize the model for this run
    Efficientnet()
