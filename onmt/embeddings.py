""" Embeddings module """
import torch
import torch.nn as nn

class Embeddings(nn.Module):
  def __init__(self, word_vec_size,
               word_vocab_size,
               word_padding_idx,
               dropout=0,
               sparse=False):
    super(Embeddings, self).__init__()
    self.word_padding_idx = word_padding_idx
    self.word_vec_size = word_vec_size
    # 训练参数encoder.embeddings.make_embedding.word.weight
    self.embedding = nn.Embedding(word_vocab_size, word_vec_size, padding_idx=word_padding_idx, sparse=sparse)
    self.embedding_size = word_vec_size

  def forward(self, source):
    source = self.embedding(source)
    return source

  # 最好使用_
  @property
  def word_lut(self):
    print("embedding.py word_lut error!")
    exit(0)
    # """ word look-up table """
    # return self.embedding

  def load_pretrained_vectors(self, emb_file, fixed):
    print("embedding.py load_pretrained_vectors error!")
    exit(0)
  #   """Load in pretrained embeddings.

  #   Args:
  #     emb_file (str) : path to torch serialized embeddings
  #     fixed (bool) : if true, embeddings are not updated
  #   """
  #   if emb_file:
  #     pretrained = torch.load(emb_file)
  #     pretrained_vec_size = pretrained.size(1)
  #     if self.word_vec_size > pretrained_vec_size:
  #       self.word_lut.weight.data[:, :pretrained_vec_size] = pretrained
  #     elif self.word_vec_size < pretrained_vec_size:
  #       self.word_lut.weight.data \
  #           .copy_(pretrained[:, :self.word_vec_size])
  #     else:
  #       self.word_lut.weight.data.copy_(pretrained)
  #     if fixed:
  #       self.word_lut.weight.requires_grad = False