
import io
from PIL import Image
import torch
import torchmetrics as tm
from torchvision import transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class IntHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text


def confusion_matrix_to_img(computed_confusion):
    # confusion matrix
    df_cm = pd.DataFrame(
        computed_confusion,
        index=self._label_ind_by_names.values(),
        columns=self._label_ind_by_names.values(),
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.05, right=.65)
    df_cm.style.background_gradient(cmap='coolwarm')
    # sn.set(font_scale=1.2)
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
    ax.legend(
        self._label_ind_by_names.values(),
        self._label_ind_by_names.keys(),
        handler_map={int: IntHandler()},
        loc='upper left',
        bbox_to_anchor=(1.2, 1)
    )
    buf = io.BytesIO()

    plt.savefig(buf, format='jpeg', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img = transforms.ToTensor()(img)
    return img


def confusion_matrix_log(self, outputs, labels, stage):
    # see https://github.com/Lightning-AI/metrics/blob/ff61c482e5157b43e647565fa0020a4ead6e9d61/docs/source/pages/lightning.rst
    # each forward pass, thus leading to wrong accumulation. In practice do the following:
    self.conf_matrix(outputs, labels)
    computed_confusion = self.conf_matrix.compute().detach().cpu().numpy().astype(int)
    confusion_mat_img = confusion_matrix_to_img(computed_confusion)
    self.logger.experiment.add_image(f'{stage}_confusion_matrix', confusion_mat_img, global_step=self.current_epoch)


def log_metrics(self, epoch_output, stage='val'):
    outputs = torch.stack([x['outputs'] for x in epoch_output])
    labels = torch.stack([x['labels'] for x in epoch_output])
    loss = torch.stack([x['loss'] for x in epoch_output]).mean()
    # _, outputs = torch.max(outputs, 1)
    # corrects = torch.sum(preds == labels.data) / len(preds)
    # Metrics
    if stage != 'train':
        confusion_matrix_log(self, outputs, labels, stage)
    self.test_auc(outputs, labels.to(torch.uint8))
    self.test_prec(outputs, labels.to(torch.uint8))
    self.test_recall(outputs, labels.to(torch.uint8))
    self.test_f1(outputs, labels.to(torch.uint8))
    self.test_acc(outputs, labels.to(torch.uint8))

    # Metrics Log
    self.log(f'{stage}_avg_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log(f'{stage}_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log(f'{stage}_f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log(f'{stage}_auc', self.test_auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log(f'{stage}_prec', self.test_prec, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log(f'{stage}_recall', self.test_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    return {}
