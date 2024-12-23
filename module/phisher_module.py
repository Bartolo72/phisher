import torchmetrics
import torch
import torch.nn.functional as F
import pytorch_lightning as pl



class PhisherhModule(pl.LightningModule):
    def __init__(self, model, optimizer, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.model = model
        self.optimizer = optimizer

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=self.num_classes)
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=self.num_classes)


    def forward(self, x):
        return self.model(x)


    def compute_loss(self, x, y):
        return F.cross_entropy(x, y)


    def common_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.compute_loss(outputs,y)
        return loss, outputs, y


    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        preds = torch.argmax(outputs, dim=1)

        acc = self.accuracy(preds, y)
        f1 = self.f1(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        return loss, acc, f1, precision, recall


    def training_step(self, batch, batch_idx):
        loss, acc, _, __, ___ = self.common_test_valid_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, acc, f1, precision, recall = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        return loss


    def test_step(self, batch, batch_idx):
        loss, acc, f1, precision, recall = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_f1', f1, prog_bar=True)
        self.log('test_precision', precision, prog_bar=True)
        self.log('test_recall', recall, prog_bar=True)
        return loss


    def configure_optimizers(self):
        return self.optimizer