
import torch
import torch.nn as nn


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        # KLDivLoss is -sum(p(x) * logq(x)), p is true dist and q is estimated dist
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """

        Args:
            x (tensor): input predict dist , of shape (N * T, V)
            target (tensor): target label, of shape (N * T)

        Returns:
            tensor, scalar:  loss tensor scalar
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # (size-2): first and last is padding
        true_dist.fill_(self.smoothing / (self.size - 2))
        idx = target.data.unsqueeze(1)
        true_dist.scatter_(1, idx, self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        # print('x:', x,  '\n true_dist:', true_dist)
        return self.criterion(x, true_dist.clone().detach())
    

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        # view x from (N, T, V ) -> (N * T, V), y (N, T) -> (N * T)
        xt = x.contiguous().view(-1, x.size(-1))
        yt = y.contiguous().view(-1)
        sloss = self.criterion(xt, yt) / norm
        return sloss.data * norm, sloss
          

def show_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    print('predict:', predict)
    print('predict log:', predict.log())
    # out = crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
    out = crit(predict, target=torch.LongTensor([2, 1, 0, 3, 3]))
    print('out:', out)
    
    
if __name__ == '__main__':
    print('test test')
    show_label_smoothing()