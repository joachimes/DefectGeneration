import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import models
from pytorch_lightning import LightningModule

class Efficientnet(LightningModule):
    def __init__(self, num_classes=14, feature_extract=False, **kwargs) -> None:
        super(Efficientnet, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.model, self.input_size = self.initialize_model(num_classes, feature_extract)


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


    def train_step(self, batch, batch_idx):
        inputs, category_labels, _ = batch

        outputs = self.model(inputs)
        loss = self.loss(outputs, category_labels)

        _, preds = torch.max(outputs, 1)
        corrects = torch.sum(preds == category_labels.data) / len(batch[0])

        return {'loss': loss, 'corrects': corrects}


    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['corrects'] for x in outputs]).mean()
        res = {'train_avg_loss': avg_loss, 'train_avg_acc': avg_acc}
        return res
        

    def val_step(self, batch, batch_idx):
        res = self.train_step(batch, batch_idx)
        return {'val_loss': res['loss'], 'val_acc': res['corrects']}


    def val_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.FloatTensor([x['val_acc'] for x in outputs]).mean()
        res = {'val_avg_loss': avg_loss, 'val_avg_acc': avg_acc}
        return res


    def test_step(self, batch, batch_idx):
        res = self.train_step(batch, batch_idx)
        return {'test_loss': res['loss'], 'test_acc': res['corrects']}


    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.FloatTensor([x['test_acc'] for x in outputs]).mean()
        res = {'test_avg_loss': avg_loss, 'test_avg_acc': avg_acc}
        return res


    def optimizer(self, parameters, lr, weight_decay):
        return Adam(parameters, lr=lr, weight_decay=weight_decay)



if __name__ == '__main__':
    # Initialize the model for this run
    Efficientnet()
