"""Base class for encoders and generic multi encoders."""

import torch.nn as nn
import onmt
from utils.misc import aeq
from onmt.sublayer import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
  def __init__(self, d_model, heads, d_ff, dropout):
    super(TransformerEncoderLayer, self).__init__()

    self.self_attn = onmt.sublayer.MultiHeadedAttention(
        heads, d_model, dropout=dropout)
    
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    # 这是自己加的LayerNorm？
    # 2组可训练参数
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.dropout = nn.Dropout(dropout)

  def forward(self, inputs, mask):
    # 这是多加的一层
    # inputs: (batch_size, s, emb+pos)
    # mask : (batch_size, 1, s)
    input_norm = self.layer_norm(inputs)
    context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                mask=mask)
    out = self.dropout(context) + inputs
    return self.feed_forward(out)


class TransformerEncoder(nn.Module):

  def __init__(self, num_layers, d_model, heads, d_ff,
               dropout, embeddings, pos):
    super(TransformerEncoder, self).__init__()

    self.num_layers = num_layers
    self.embeddings = embeddings
    self.postion = pos
    # num_layers的参数不共享
    self.transformer = nn.ModuleList(
      [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
       for _ in range(num_layers)])
    # nn.LayerNorm输入与输出的尺寸相同
    # 2组可训练参数
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

  def _check_args(self, src, lengths=None):
    _, n_batch = src.size()
    if lengths is not None:
      n_batch_, = lengths.size()
      aeq(n_batch, n_batch_)

  def forward(self, src, lengths=None):
    """ See :obj:`EncoderBase.forward()`"""
    # 确定src是否是二维数组, sentence_length * batch_size(截断后)
    self._check_args(src, lengths)

    # 此处的emb其实是nn.Embedding的参数(weight)+固定的位置参数所得,
    # 可变的是nn.Embedding的参数
    emb = self.embeddings(src)
    if self.postion is not None:
      emb = self.postion(emb)

    # (s, batch_size, emb+pos)  ==> (batch_size, s, emb+pos)
    # 必须调用contiguous
    out = emb.transpose(0, 1).contiguous()
    # (s, batch_size)
    words = src.transpose(0, 1)
    # <blank>
    padding_idx = self.embeddings.word_padding_idx
    # mask掉填充字符
    mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
    # Run the forward pass of every layer of the tranformer.
    for i in range(self.num_layers):
      out = self.transformer[i](out, mask)
    # batch_size, s, dim
    out = self.layer_norm(out)
    # 返回(s, batch_size, dim)
    return emb, out.transpose(0, 1).contiguous(), lengths

