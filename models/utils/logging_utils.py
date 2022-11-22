
import io
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import seaborn as sn
import torchmetrics as tm
import pandas as pd
import matplotlib.pyplot as plt



class IntHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text


def confusion_matrix_to_img(class_names, computed_confusion):
    # confusion matrix
    df_cm = pd.DataFrame(
        computed_confusion,
        index=class_names.keys(),
        columns=class_names.keys(),
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.05, right=.65)
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
    ax.legend(
        class_names.values(),
        class_names.keys(),
        handler_map={int: IntHandler()},
        loc='upper left',
        bbox_to_anchor=(1.2, 1)
    )
    buf = io.BytesIO()

    plt.savefig(buf, format='jpeg', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img = transforms.ToTensor()(img)
    #close figure
    plt.close()
    return img


def confusion_matrix_log(self, outputs, labels, stage):
    # see https://github.com/Lightning-AI/metrics/blob/ff61c482e5157b43e647565fa0020a4ead6e9d61/docs/source/pages/lightning.rst
    # each forward pass, thus leading to wrong accumulation. In practice do the following:
    conf_matrix = tm.ConfusionMatrix(num_classes=self.num_classes).to(self.device)
    
    conf_matrix(outputs, labels)
    computed_confusion = conf_matrix.compute().detach().cpu().numpy().astype(int)
    confusion_mat_img = confusion_matrix_to_img(self.class_names, computed_confusion)
    self.logger.experiment.add_image(f'{stage}_confusion_matrix', confusion_mat_img, global_step=self.current_epoch)


def log_metrics(self, epoch_output, stage='val'):
    outputs = torch.cat([x['outputs'] for x in epoch_output], dim=0)
    probs = F.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)
    labels = torch.cat([x['labels'] for x in epoch_output], dim=0)
    loss = torch.stack([x['loss'] for x in epoch_output]).mean()
    # Metrics
    if stage != 'train':
        self.test_acc(preds, labels.to(torch.uint8))
        confusion_matrix_log(self, preds, labels, stage)
        self.test_prec(preds, labels.to(torch.uint8))
        self.test_recall(preds, labels.to(torch.uint8))
        self.test_f1(preds, labels.to(torch.uint8))
        # self.test_auc(probs, labels.to(torch.uint8))
        # Metrics Log
        self.log(f'{stage}_acc', self.test_acc, on_epoch=True, logger=True)
        self.log(f'{stage}_f1', self.test_f1, on_epoch=True, logger=True)
        # self.log(f'{stage}_auc', self.test_auc, on_epoch=True, logger=True)
        self.log(f'{stage}_prec', self.test_prec, on_epoch=True, logger=True)
        self.log(f'{stage}_recall', self.test_recall, on_epoch=True, logger=True)
    return {f'{stage}_avg_loss': loss}
