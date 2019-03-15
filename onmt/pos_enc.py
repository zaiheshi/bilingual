import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
  def __init__(self, dropout, dim, max_len=5000):
    super(PositionalEncoding, self).__init__()
    # dim是embedding_size
    # 将位置p映射为一个dim(d_model)维的位置向量
    # max_len必须不小于word_vocab_size
    pe = torch.zeros(max_len, dim)
    # torch.arange(0, max_len) ==> tensor([0,1, 2, ... , max_len-1])
    # unsqueeze增加维度但是共享内存, size: (max_len, 1)
    position = torch.arange(0, max_len).unsqueeze(1)
    # 注意torch.float
    # https://www.jiqizhixin.com/articles/2018-01-10-20
    # 2i只与dim有关且2i < dim, 可以单独拿出，arange取出所有的数值[0, 2,..., 512]
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                         -(math.log(10000.0) / dim)))
    # 得到max_len * 256的数组
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    # (max_len, 1, dim), 词嵌入后大小为xx * batch_size * dim, 
    # forward中截断max_len为batch_size, 加法这里不太懂，只知道是兼容的
    pe = pe.unsqueeze(1)
    # 这块缓冲区的作用？ 不让pe的参数在反向传播过程中被更新
    # 下面代码执行后，pe会成为self的属性
    # 原因是__getattr_
    self.register_buffer('pe', pe)
    self.dropout = nn.Dropout(p=dropout)
    self.dim = dim

  def forward(self, emb, step=None):
    # 为什么要乘以根号512?
    emb = emb * math.sqrt(self.dim)
    if step is None:
      # 截断相加
      emb = emb + self.pe[:emb.size(0)]
    else:
      emb = emb + self.pe[step]
    # 注意dropout
    emb = self.dropout(emb)
    return emb