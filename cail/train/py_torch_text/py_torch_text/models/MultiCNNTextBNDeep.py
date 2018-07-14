from .BasicModule import BasicModule
import torch as t
import numpy as np
from torch import nn
import pdb

kernel_sizes =  [1,2,3,4]
kernel_sizes2 = [1,2,3,4]
class MultiCNNTextBNDeep(BasicModule): 
    def __init__(self, opt ):
        super(MultiCNNTextBNDeep, self).__init__()
        self.model_name = 'MultiCNNTextBNDeep'
        self.opt=opt
        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)

        data_convs = [ nn.Sequential(
                                nn.Conv1d(in_channels = opt.embedding_dim,
                                        out_channels = opt.data_dim,
                                        kernel_size = kernel_size),
                                nn.BatchNorm1d(opt.data_dim),
                                #nn.Dropout(0.5),
                                nn.ReLU(inplace=True),
                                nn.Conv1d(in_channels = opt.data_dim,
                                        out_channels = opt.data_dim,
                                        kernel_size = kernel_size),
                                nn.BatchNorm1d(opt.data_dim),
                                #nn.Dropout(0.5),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(kernel_size = (opt.data_seq_len - kernel_size*2 + 2))
                            )
         for kernel_size in kernel_sizes]


        self.data_convs = nn.ModuleList(data_convs)

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes)*opt.data_dim,opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )

        if opt.embedding_path:
            self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))

    def forward(self, data):
        en_data = self.encoder(data)

        if self.opt.static:
            en_data.detach()

        data_out = [data_conv(en_data.permute(0, 2, 1)) for data_conv in self.data_convs]
        conv_out = t.cat(data_out,dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits

    # def get_optimizer(self):  
    #    return  t.optim.Adam([
    #             {'params': self.data_conv.parameters()},
    #             {'params': self.content_conv.parameters()},
    #             {'params': self.fc.parameters()},
    #             {'params': self.encoder.parameters(), 'lr': 5e-4}
    #         ], lr=self.opt.lr)
    # # end method forward


 
if __name__ == '__main__':
    from ..config import opt
    opt.data_dim = 100
    m = MultiCNNText(opt)
    data = t.autograd.Variable(t.arange(0,500).view(10,50)).long()
    content = t.autograd.Variable(t.arange(0,2500).view(10,250)).long()
    o = m(data,content)
    print(o.size())
