"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np

import onmt
from onmt.sublayer import PositionwiseFeedForward

MAX_SIZE = 5000


class TransformerDecoderLayer(nn.Module):
  def __init__(self, d_model, heads, d_ff, dropout):
    super(TransformerDecoderLayer, self).__init__()

    self.self_attn = onmt.sublayer.MultiHeadedAttention(
      heads, d_model, dropout=dropout)

    self.context_attn = onmt.sublayer.MultiHeadedAttention(
      heads, d_model, dropout=dropout)
    
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
    self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
    
    self.dropout = dropout
    self.drop = nn.Dropout(dropout)
    mask = self._get_attn_subsequent_mask(MAX_SIZE)
    # Register self.mask as a buffer in TransformerDecoderLayer, so
    # it gets TransformerDecoderLayer's cuda behavior automatically.
    # register_buffer查资料
    self.register_buffer('mask', mask)

  def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
              layer_cache=None, step=None):
    dec_mask = None
    # (64, 1, 20) + (1, 20, 20) = (64, 20, 20)
    if step is None:
      # tgt_pad_mask是否需要加, 需要，一句话很短的时候需要
      dec_mask = torch.gt(tgt_pad_mask +
                          self.mask[:, :tgt_pad_mask.size(-1),
                                    :tgt_pad_mask.size(-1)], 0)

    input_norm = self.layer_norm_1(inputs)

    # mask的运用
    query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=dec_mask,
                                 layer_cache=layer_cache,
                                 type="self")

    query = self.drop(query) + inputs

    query_norm = self.layer_norm_2(query)
    mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                  mask=src_pad_mask,
                                  layer_cache=layer_cache,
                                  type="context")
    output = self.feed_forward(self.drop(mid) + query)

    return output, attn

  def _get_attn_subsequent_mask(self, size):
    attn_shape = (1, size, size)
    # 上三角，不填充对角线1
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask


class TransformerDecoder(nn.Module):
  def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, pos):
    super(TransformerDecoder, self).__init__()

    # Basic attributes.
    self.decoder_type = 'transformer'
    self.num_layers = num_layers
    self.embeddings = embeddings
    self.position = pos

    # Decoder State
    self.state = {}

    # Build TransformerDecoder.
    # num_layers = 2
    self.transformer_layers_fw = nn.ModuleList(
      [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
       for _ in range(num_layers)])
    self.transformer_layers_bw = nn.ModuleList(
      [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
       for _ in range(num_layers)])

    self.layer_norm_fw = nn.LayerNorm(d_model, eps=1e-6)
    self.layer_norm_bw = nn.LayerNorm(d_model, eps=1e-6)

  def init_state(self, src, src_enc):
    """ Init decoder state """
    self.state["src"] = src
    self.state["src_enc"] = src_enc
    self.state["cache"] = None

  def map_state(self, fn):
    def _recursive_map(struct, batch_dim=0):
      for k, v in struct.items():
        if v is not None:
          if isinstance(v, dict):
            _recursive_map(v)
          else:
            struct[k] = fn(v, batch_dim)

    self.state["src"] = fn(self.state["src"], 1)
    self.state["src_enc"] = fn(self.state["src_enc"], 1)
    if self.state["cache"] is not None:
      _recursive_map(self.state["cache"])

  def detach_state(self):
    self.state["src"] = self.state["src"].detach()

  def forward(self, tgt, step=None):
    """
    See :obj:`onmt.modules.RNNDecoderBase.forward()`
    """
    if step == 0:
      self._init_cache(self.num_layers)

    # xx * batch_size
    src = self.state["src"]
    # xx * batch_size * yy
    memory_bank = self.state["src_enc"]
    # xx * batch_size ==> batch_size * xx    

    # src用在这里
    src_words = src.transpose(0, 1)
    tgt_words = tgt.transpose(0, 1)

    # Initialize return variables.
    attns = {"std": []}

    # Run the forward pass of the TransformerDecoder.
    # xx * batch_size * dim
    emb_fw = self.embeddings(tgt)
    # flip不行, 分片不支持, 只能改tgt了
    emb_bw = self.embeddings(torch.flip(tgt, (0,)))
    if self.position is not None:
      emb_fw = self.position(emb_fw)
      emb_bw = self.position(emb_bw)
    assert emb_fw.dim() == 3  # len x batch x embedding_dim
    assert emb_bw.dim() == 3  # len x batch x embedding_dim

    # 模型之间的参数是互不干扰的, 不能让下一个神经网络改变
    # 当前层的输出
    # output:  batch_size * xx * dim
    # transpose, view, ...调用之后一般需要调用contiguous来返回一个拷贝
    output_fw = emb_fw.transpose(0, 1).contiguous()
    output_bw = emb_bw.transpose(0, 1).contiguous()
    # src_memory_bank: batch_size * xx * dim, encoder层的输出结果
    src_memory_bank = memory_bank.transpose(0, 1).contiguous()
    pad_idx = self.embeddings.word_padding_idx
    # 找出src与tgt中所有填充的字符全部置为1
    src_pad_mask = src_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_src]
    tgt_pad_mask_fw = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]
    # 填充字符也需要改变
    tgt_pad_mask_bw = torch.flip(tgt_words,(1,)).data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

    # output是来自词嵌入后的tgt, batch_size * xx * dim
    # src_memory_bank是encoder层的输出，batch_size * xx * dim 
    for i in range(self.num_layers):
      output_fw, _ = self.transformer_layers_fw[i](
        output_fw,
        src_memory_bank,
        src_pad_mask,
        tgt_pad_mask_fw,
        layer_cache=(
          self.state["cache"]["layer_{}".format(i)]
          if step is not None else None),
        step=step)
      output_bw, _ = self.transformer_layers_bw[i](
        output_bw,
        src_memory_bank,
        src_pad_mask,
        tgt_pad_mask_bw,
        layer_cache=(
          self.state["cache"]["layer_{}".format(i)]
          if step is not None else None),
        step=step)
    # output: (64, 16, 512) attn: (64, 13, 11)
    output_fw = self.layer_norm_fw(output_fw)
    output_bw = self.layer_norm_bw(output_fw)

    # Process the result and update the attentions.    
    # xx * batch_size * dim
    dec_outs_fw = output_fw.transpose(0, 1).contiguous()
    dec_outs_bw = output_bw.transpose(0, 1).contiguous()

    # attn_fw = attn_fw.transpose(0, 1).contiguous()
    # attn_bw = attn_bw.transpose(0, 1).contiguous()

    # attns["std_fw"] = attn_fw
    # attns["std_bw"] = attn_bw

    # TODO change the way attns is returned dict => list or tuple (onnx)
    return emb_fw, dec_outs_fw, dec_outs_bw

  def _init_cache(self, num_layers):
    self.state["cache"] = {}

    for l in range(num_layers):
      layer_cache = {
        "memory_keys": None,
        "memory_values": None
      }
      layer_cache["self_keys"] = None
      layer_cache["self_values"] = None
      self.state["cache"]["layer_{}".format(l)] = layer_cache
