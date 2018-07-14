from .BasicModule import BasicModule
import torch as t
import numpy as np
from torch import nn


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

class LSTMText(BasicModule):
    def __init__(self, opt):
        super(LSTMText, self).__init__()
        self.model_name = 'LSTMText'
        self.opt=opt

        kernel_size = opt.kernel_size
        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)

        self.data_lstm =nn.LSTM(input_size = opt.embedding_dim,\
                            hidden_size = opt.hidden_size,
                            num_layers = opt.num_layers,
                            bias = True,
                            batch_first = False,
                            # dropout = 0.5,
                            bidirectional = True
                            )

        # self.dropout = nn.Dropout()
        self.fc = nn.Sequential(
            nn.Linear(opt.kmax_pooling*(opt.hidden_size*2*1),opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )
        # self.fc = nn.Linear(3 * (opt.title_dim+opt.content_dim), opt.num_classes)
        if opt.embedding_path:
            self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))
 
    def forward(self, data):
        data = self.encoder(data)
        if self.opt.static:
            data=data.detach()
        
        data_out = self.data_lstm(data.permute(1,0,2))[0].permute(1,2,0)

        conv_out = kmax_pooling((data_out),2,self.opt.kmax_pooling)

        reshaped = conv_out.view(conv_out.size(0), -1)

        logits = self.fc((reshaped))
        return logits

    # def get_optimizer(self):  
    #    return  t.optim.Adam([
    #             {'params': self.title_conv.parameters()},
    #             {'params': self.content_conv.parameters()},
    #             {'params': self.fc.parameters()},
    #             {'params': self.encoder.parameters(), 'lr': 5e-4}
    #         ], lr=self.opt.lr)
    # # end method forward

