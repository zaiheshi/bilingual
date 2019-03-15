"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.constants as Constants 

from onmt.transformer_encoder import TransformerEncoder
from onmt.transformer_decoder import TransformerDecoder

from onmt.embeddings import Embeddings
from onmt.pos_enc import PositionalEncoding
from utils.misc import use_gpu
from utils.logging import logger
from inputters.dataset import load_fields_from_vocab

class NMTModel(nn.Module):
  def __init__(self, encoder, decoder, model_opt):
    super(NMTModel, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.dense = nn.Linear(2*model_opt.dec_rnn_size, 2*model_opt.dec_rnn_size, bias=False)
    # self.output_layer = nn.Linear(4*model_opt.dec_rnn_size, model_opt.vocab_size_tgt, bias=False)
  def shift_concat(self, fw_outputs, bw_outputs):
    fw_ft = torch.zeros_like(fw_outputs[:, 0:1])
    bw_ft = torch.zeros_like(bw_outputs[:, 0:1])
    fw_outputs = torch.cat([fw_ft, fw_outputs[:, :-1]], 1)
    bw_outputs = torch.cat([bw_outputs[:, 1:], bw_ft], 1)
    return torch.cat([fw_outputs, bw_outputs], -1)
  def forward(self, src, tgt, lengths):
    # src的数据结尾全部是2, tgt的开头是2, 末尾是3, 填充字符是1
    # tgt = tgt[:-1]  # exclude last target from inputs
    _, memory_bank, lengths = self.encoder(src, lengths)
    # src是用来在decoder层的第二个multiply-head-attention上做mask操作
    # decoder层的自注意力是用tgt和其他一些来做mask操作
    self.decoder.init_state(src, memory_bank)
    # xx * batch_size * dim
    emb_fw, dec_out_fw, dec_out_bw = self.decoder(tgt)

    dec_out_fw = dec_out_fw.transpose(0, 1).contiguous()
    dec_out_bw = dec_out_bw.transpose(0, 1).contiguous()
    emb_fw = emb_fw.transpose(0, 1).contiguous()

    shift_outputs = self.shift_concat(dec_out_fw, dec_out_bw)
    shift_inputs = self.shift_concat(emb_fw, emb_fw)
    shift_proj_inputs = self.dense(shift_inputs)
    # batch_size * len * 2048
    bi_outputs = torch.cat([shift_outputs, shift_proj_inputs], -1)
    # batch_size * len * vocab_size
    # logits = self.output_layer(bi_outputs)
    # 本来返回的是decoder端的输出与注意力
    bi_outputs = bi_outputs.transpose(0,1).contiguous()
    return bi_outputs

def build_embeddings(opt, word_dict, for_encoder=True):
  """
  Build an Embeddings instance.
  Args:
      opt: the option in current environment.
      word_dict(Vocab): words dictionary.
      feature_dicts([Vocab], optional): a list of feature dictionary.
      for_encoder(bool): build Embeddings for encoder or decoder?
  """
  if for_encoder:
    # word_embedding大小默认是512
    embedding_dim = opt.src_word_vec_size
  else:
    embedding_dim = opt.tgt_word_vec_size

  # 获取填充数值 1
  word_padding_idx = word_dict.stoi[Constants.PAD_WORD]
  # 获取vocab大小
  num_word_embeddings = len(word_dict)

  return Embeddings(word_vec_size=embedding_dim,
                    dropout=opt.dropout,
                    word_padding_idx=word_padding_idx,
                    word_vocab_size=num_word_embeddings,
                    sparse=opt.optim == "sparseadam")

def build_position_encoding(opt, for_encoder = True):
  if for_encoder:
    embedding_dim = opt.src_word_vec_size
  else:
    embedding_dim = opt.tgt_word_vec_size
  return PositionalEncoding(opt.dropout, embedding_dim)

def build_encoder(opt, embeddings, pos = None):
  """
  Various encoder dispatcher function.
  Args:
      opt: the option in current environment.
      embeddings (Embeddings): vocab embeddings for this encoder.
  """
  # 根据transformer论文，整个encoder层由enc_layers=6层堆叠而成，d_model = enc_rnn_size = 512,
  # multi-head attention中 h = heads = 8, feed-forward层大小为transformer_ff=2048 
  return TransformerEncoder(opt.enc_layers, opt.enc_rnn_size,
                            opt.heads, opt.transformer_ff,
                            opt.dropout, embeddings, pos)

def build_decoder(opt, embeddings, pos = None):
  """
  Various decoder dispatcher function.
  Args:
      opt: the option in current environment.
      embeddings (Embeddings): vocab embeddings for this decoder.
  """
  # 根据transformer论文，整个decoder层由dec_layers=6层堆叠而成，d_model = dec_rnn_size = 512,
  # multi-head attention中 h = heads = 8, feed-forward层大小为transformer_ff=2048 
  return TransformerDecoder(opt.dec_layers, opt.dec_rnn_size,
                     opt.heads, opt.transformer_ff,
                     opt.dropout, embeddings, pos)

def load_test_model(opt, dummy_opt, model_path=None):
  if model_path is None:
    # 仅载入一个模型文件
    model_path = opt.models[0]
  checkpoint = torch.load(model_path,
                        map_location=lambda storage, loc: storage)
  fields = load_fields_from_vocab(checkpoint['vocab'])

  model_opt = checkpoint['opt']

  # 使得model_opt包含dummy_opt的信息
  for arg in dummy_opt:
    if arg not in model_opt:
      model_opt.__dict__[arg] = dummy_opt[arg]
  model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
  model.eval()
  # 为什么需要单独拿出?
  model.generator.eval()
  return fields, model


def build_base_model(model_opt, fields, gpu, checkpoint=None):
  """
  Args:
      # 模型参数
      model_opt: the option loaded from checkpoint.
      # vocabuary table, "src" and "tgt"
      fields: `Field` objects for the model.
      gpu(bool): whether to use gpu.
      checkpoint: the model gnerated by train phase, or a resumed snapshot
                  model from a stopped training.
  Returns:
      the NMTModel.
  """
  # for backward compatibility
  # 512
  # Size of encoder rnn hidden states.
  # Size of decoder rnn hidden states.
  # if model_opt.enc_rnn_size != model_opt.dec_rnn_size:
  #   raise AssertionError("""We do not support different encoder and
  #                        decoder rnn sizes for translation now.""")
  # Build encoder.
  # src中的词表
  src_dict = fields["src"].vocab
  tgt_dict = fields["tgt"].vocab
  # 构建embedding需要必要的参数以及vocab
  # 结果包含word_embedding与position_encoding两个模块以及word_embedding大小
  # 和填充数值("blank", 1)
  model_opt.vocab_size_src = len(src_dict)
  model_opt.vocab_size_tgt = len(tgt_dict)

  src_embeddings = build_embeddings(model_opt, src_dict)

  assert(model_opt.position_encoding == True)

  if model_opt.position_encoding:
    src_pos_enc = build_position_encoding(model_opt)
  else:
    src_pos_enc = None
  encoder = build_encoder(model_opt, src_embeddings, src_pos_enc)

  # Build decoder.
  tgt_embeddings = build_embeddings(model_opt, tgt_dict, for_encoder=False)
  if model_opt.position_encoding:
    tgt_pos_enc = build_position_encoding(model_opt, for_encoder=False)
  else:
    tgt_pos_enc = None
  decoder = build_decoder(model_opt, tgt_embeddings, tgt_pos_enc)

  # Build NMTModel(= encoder + decoder).
  # 第一次出现
  device = torch.device("cuda:0" if gpu else "cpu")
  model = NMTModel(encoder, decoder, model_opt)

  # Build Generator.
  gen_func = nn.LogSoftmax(dim=-1)
  # 输出每个词的概率
  generator = nn.Sequential(
    nn.Linear(4 * model_opt.dec_rnn_size, len(fields["tgt"].vocab)),
    gen_func
  )
  # if model_opt.share_decoder_embeddings:
  #   generator[0].weight = decoder.embeddings.word_lut.weight

  # Load the model states from checkpoint or initialize them.
  if checkpoint is not None:
    # This preserves backward-compat for models using customed layernorm
    def fix_key(s):
      s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                 r'\1.layer_norm\2.bias', s)
      s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                 r'\1.layer_norm\2.weight', s)
      return s

    checkpoint['model'] = \
      {fix_key(k): v for (k, v) in checkpoint['model'].items()}
    # end of patch for backward compatibility
    model.load_state_dict(checkpoint['model'], strict=False)
    generator.load_state_dict(checkpoint['generator'], strict=False)
  else:
    # 然而参数就是0.0
    if model_opt.param_init != 0.0:
      for p in model.parameters():
        p.data.uniform_(-model_opt.param_init, model_opt.param_init)
      for p in generator.parameters():
        p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    # true
    if model_opt.param_init_glorot:
      for p in model.parameters():
        if p.dim() > 1:
          xavier_uniform_(p)
      for p in generator.parameters():
        if p.dim() > 1:
          xavier_uniform_(p)

    # if hasattr(model.encoder, 'embeddings'):
    #   model.encoder.embeddings.load_pretrained_vectors(
    #       model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
    # if hasattr(model.decoder, 'embeddings'):
    #   model.decoder.embeddings.load_pretrained_vectors(
    #       model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)
  #pdb.set_trace()
  # Add generator to model (this registers it as parameter of model).
  model.generator = generator
  model.to(device)
  return model


def build_model(model_opt, opt, fields, checkpoint):
  """ Build the Model """
  logger.info('Building model...')
  model = build_base_model(model_opt, fields,
                           use_gpu(opt), checkpoint)
  logger.info(model)
  return model
