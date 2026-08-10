import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class Isic19Loss(_Loss):
    """Weighted focal loss for ISIC19."""

    def __init__(
        self,
        alpha=torch.tensor(
            [5.5813, 2.0472, 7.0204, 26.1194,
             9.5369, 101.0707, 92.5224, 38.3443]
        ),
        gamma=2.0,
    ):
        super().__init__()
        self.alpha = alpha.to(torch.float)
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets = targets.view(-1, 1).type_as(inputs)

        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.long())
        logpt = logpt.view(-1)

        pt = logpt.exp()

        alpha = self.alpha.to(targets.device)
        at = alpha.gather(0, targets.data.view(-1).long())

        logpt = logpt * at
        loss = -(1 - pt) ** self.gamma * logpt

        return loss.mean()


def loss_fn_loading(dataset="camelyon17"):
    match dataset:
        case "isic19":
            return Isic19Loss()
        case "ultrasound" | "xray":
            return nn.BCEWithLogitsLoss()
        case _:
            return nn.CrossEntropyLoss()