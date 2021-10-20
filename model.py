
import torch
from torch import nn
from torch.nn import functional as F


class autoEncoder(nn.Module):
    def __init__(self, input_dim, hidden):
        super(autoEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden if isinstance(hidden, list) else [hidden]
        self.input_hidden = [input_dim] + self.hidden

        self.linears_encoder = nn.ModuleList(
            [nn.Linear(self.input_hidden[i], self.input_hidden[i+1]) for i in range(len(self.input_hidden)-1)]
        )
        self.linears_decoder = nn.ModuleList(
            [nn.Linear(self.input_hidden[i], self.input_hidden[i-1]) for i in range(len(self.input_hidden)-1, 0, -1)][::-1]
        )

        self.batchNorms = nn.ModuleList(
            [nn.BatchNorm1d(i, momentum=0.1, affine=True) for i in self.input_hidden]
        )


    def encoder(self, x, idx_layer):
        x = self.linears_encoder[idx_layer](x)
        x = self.batchNorms[idx_layer+1](x)
        x = torch.sigmoid(x)

        return x

    def decoder(self, y, idx_layer):
        y = self.linears_decoder[idx_layer](y)
        y = self.batchNorms[idx_layer](y)
        y = torch.sigmoid(y)

        return y

    def decoder_end(self, y, idx_layer):
        for i in range(idx_layer, -1, -1):
            y = self.decoder(y, i)

        return y

    def forward(self, x, idx_layer, flag='training'):
        encoder_op = self.encoder(x, idx_layer)

        if flag == 'training':
            decoder_op = self.decoder(encoder_op, idx_layer)
        else:
            decoder_op = self.decoder_end(encoder_op, idx_layer)

        return decoder_op, encoder_op


class encoderLoss(nn.Module):
    def __init__(self, mustlinks, batch_size,
                 gamma):
        super(encoderLoss, self).__init__()
        self._mustlinks = mustlinks
        self._batch_size = batch_size
        self._gamma = gamma
        self.mse = torch.nn.MSELoss(reduction='sum')

    def forward(self, y_pred, y_truth, data_indx):
        M = self._mustlinks[data_indx][:, data_indx].float()
        loss_ml = self.get_constraints_loss(y_pred, M)
        # loss = F.binary_cross_entropy(y_pred, y_truth)
        loss = F.mse_loss(y_pred, y_truth)

        return loss + self._gamma * loss_ml, loss_ml

    def get_constraints_loss(self, y_pred, constraints_batch):
        D = torch.diag(constraints_batch.sum(axis=1))
        L = D - constraints_batch
        loss = torch.trace(torch.matmul(torch.matmul(y_pred.T, L), y_pred)) * 2

        return loss / (self._batch_size * self._batch_size)
