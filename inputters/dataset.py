from itertools import chain
from torchtext.data.utils import RandomShuffler
import gc
import glob
import codecs
from collections import defaultdict

import torch
import torchtext.data
from utils.logging import logger
import onmt.constants as Constants

def _getstate(self):
  return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
  self.__dict__.update(state)
  self.stoi = defaultdict(lambda: 0, self.stoi)

torchtext.vocab.Vocab.__getstate__ = _getstate
torchtext.vocab.Vocab.__setstate__ = _setstate


def make_text_iterator_from_file(path):
  with codecs.open(path, "r", "utf-8") as corpus_file:
    for line in corpus_file:
      # 迭代
      yield line

def make_features(batch, side):
  """
  Args:
      batch (Tensor): a batch of source or target data.
      side (str): for source or for target.
  Returns:
      A sequence of src/tgt tensors with optional feature tensors
      of size (len x batch).
  """
  assert side in ['src', 'tgt']
  if isinstance(batch.__dict__[side], tuple):
    # 如果是元组则返回第一个，第二个是长度
    data = batch.__dict__[side][0]
  else:
    data = batch.__dict__[side]

  return data

def save_fields_to_vocab(fields):
  """
  Save Vocab objects in Field objects to `vocab.pt` file.
  """
  vocab = []
  for k, f in fields.items():
    if f is not None and 'vocab' in f.__dict__:
      f.vocab.stoi = f.vocab.stoi
      vocab.append((k, f.vocab))
  return vocab

def get_source_fields(fields=None):
  if fields is None:
    fields = {}

  fields["src"] = torchtext.data.Field(
    pad_token=Constants.PAD_WORD,
    include_lengths=True)

  fields["indices"] = torchtext.data.Field(
      use_vocab=False, dtype=torch.long,
      sequential=False)

  return fields

def get_target_fields(fields=None):
  if fields is None:
    fields = {}

  fields["tgt"] = torchtext.data.Field(
    init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD,
    pad_token=Constants.PAD_WORD)

  fields["indices"] = torchtext.data.Field(
      use_vocab=False, dtype=torch.long,
      sequential=False)

  return fields

def get_fields():
  fields = {}

  # <blank>用作pad字符, 数值为0
  # 不但返回pad后的数据，还返回每批数据的长度
  fields["src"] = torchtext.data.Field(
    pad_token=Constants.PAD_WORD,
    include_lengths=True)
  # 填充字符是<blank>, 数值为0
  # init_token: <s> 数值为2
  # eos_token: </s> 数值为3
  fields["tgt"] = torchtext.data.Field(
    init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD,
    pad_token=Constants.PAD_WORD)

  # 无需多余处理
  fields["indices"] = torchtext.data.Field(
      use_vocab=False, dtype=torch.long,
      sequential=False)

  return fields
  #fields = {}
    
  #fields = get_source_fields(fields)
  #fields = get_target_fields(fields)

  #return fields

def load_fields_from_vocab(vocab):
  """
  Load Field objects from `vocab.pt` file.
  """
  # 列表转字典, 结果{"src": vocab, "tgt": vocab}
  vocab = dict(vocab)
  fields = get_fields()
  for k, v in vocab.items():
    # Hack. Can't pickle defaultdict :(
    # 访问键不存在，默认值是0. 
    v.stoi = defaultdict(lambda: 0, v.stoi)
    # 自己在类外添加的成员变量vacab <unk> = 0, <blank> = 1在src stoi与itos里面
    # <unk> <blank> <s> </s>在tgt stoi与itos里面
    fields[k].vocab = v
  return fields

def load_fields(opt, checkpoint):
  if checkpoint is not None:
    logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
    fields = load_fields_from_vocab(checkpoint['vocab'])
  else:
    fields = load_fields_from_vocab(torch.load(opt.data + '_vocab.pt'))

  logger.info(' * vocabulary size. source = %d; target = %d' %
              (len(fields['src'].vocab), len(fields['tgt'].vocab)))

  return fields

class DatasetIter(object):
  """ An Ordered Dataset Iterator, supporting multiple datasets,
      and lazy loading.

  Args:
      datsets (list): a list of datasets, which are lazily loaded.
      fields (dict): fields dict for the datasets.
      batch_size (int): batch size.
      batch_size_fn: custom batch process function.
      device: the GPU device.
      is_train (bool): train or valid?
  """

  def __init__(self, datasets, fields, batch_size, batch_size_fn,
               device, is_train):
    # 载入的数据文件
    self.datasets = datasets
    # 词表
    self.fields = fields
    self.batch_size = batch_size
    self.batch_size_fn = batch_size_fn
    self.device = device
    self.is_train = is_train

    self.cur_iter = self._next_dataset_iterator(datasets)
    # We have at least one dataset.
    assert self.cur_iter is not None

  # 没有实现__next__方法，按这种写法，只能用for循环进行迭代,且可以支持多次迭代
  # 根据目前我了解的知识，__iter__方法里面可以使用Yield，也可以返回一个可迭代对象
  def __iter__(self):
    # 此处的dateset_iter是从第二批次开始的，第一批次在创建实例时候被_next_dataset_iterator
    # 拿出了
    dataset_iter = (d for d in self.datasets)
    # 这样便可以直接for循环迭代了
    while self.cur_iter is not None:
      for batch in self.cur_iter:
        yield batch
      self.cur_iter = self._next_dataset_iterator(dataset_iter)

  def __len__(self):
    # We return the len of cur_dataset, otherwise we need to load
    # all datasets to determine the real len, which loses the benefit
    # of lazy loading.
    assert self.cur_iter is not None
    return len(self.cur_iter)

  def _next_dataset_iterator(self, dataset_iter):
    try:
      # 这个类的实例仅会创建一次, 不加下面的if代码是否会自动回收内存?
      # Drop the current dataset for decreasing memory
      if hasattr(self, "cur_dataset"):
        self.cur_dataset.examples = None
        gc.collect()
        del self.cur_dataset
        gc.collect()
      # 读入一份文件的数据保存在cur_dataset中
      self.cur_dataset = next(dataset_iter)
    except StopIteration:
      return None

    # self.cur_dataset.examples(一份src、tgt的完整内容) / self.cur_dataset.fields组成一份完整的数据文件
    # We clear `fields` when saving, restore when loading.
    self.cur_dataset.fields = self.fields

    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
    # self.cur_dataset: examples = list of Example, fields = self.fields, src_vocab = []
    
    return OrderedIterator(
      dataset=self.cur_dataset, batch_size=self.batch_size,
      batch_size_fn=self.batch_size_fn,
      device=self.device, train=self.is_train,
      sort=False, sort_within_batch=True,
      repeat=False)
    
class OrderedIterator(torchtext.data.Iterator):
  """ Ordered Iterator Class """

  # 只改变方法，不需要调用super
  def create_batches(self):
    """ Create batches """
    if self.train:
      def _pool(data, random_shuffler):
        # p: [Example,...] ,size = batch_size * 100
        # torchtext.data.Dataset里的__iter__使得for循环遍历
        # data返回所有examples

        # 首先得到batch_size为3200的数据Examples list的generator
        # data的迭代器，每次返回一个example对象
        for p in torchtext.data.batch(data, self.batch_size * 100):
          # 再得到batch_size为64的数据的generator
          p_batch = torchtext.data.batch(
            sorted(p, key=self.sort_key),
            self.batch_size, self.batch_size_fn)
          # 随机打乱100个大小为64的数据
          for b in random_shuffler(list(p_batch)):
            # 最后得到batch_size为64的数据
            yield b

      # 其上调用next将会得到batch大小为1的数据
      # 在超类中__iter__最终的返回值是
      # (batch_size*src句子, batch_size*tgt句子), 且这些句子全部处理完成，
      # 长度固定，特殊字符加入
      self.batches = _pool(self.data(), self.random_shuffler)
    else:
      self.batches = []
      for b in torchtext.data.batch(self.data(), self.batch_size,
                                    self.batch_size_fn):
        self.batches.append(sorted(b, key=self.sort_key))


def load_dataset(corpus_type, opt):
  assert corpus_type in ["train", "valid"]

  def _dataset_loader(pt_file, corpus_type):
    dataset = torch.load(pt_file)
    logger.info('Loading %s dataset from %s, number of examples: %d' %
                (corpus_type, pt_file, len(dataset)))
    return dataset

  # Sort the glob output by file name (by increasing indexes).
  pts = sorted(glob.glob(opt.data + '_' + corpus_type + '.[0-9]*.pt'))
  if pts:
    for pt in pts:
      yield _dataset_loader(pt, corpus_type)
  else:
    pt = opt.data + '_' + corpus_type + '.pt'
    yield _dataset_loader(pt, corpus_type)

def build_dataset(fields,
                  src_data_iter,
                  tgt_data_iter,
                  src_seq_length=0, tgt_seq_length=0,
                  src_seq_length_trunc=0, tgt_seq_length_trunc=0,
                  use_filter_pred=True):
  assert src_data_iter != None
  assert tgt_data_iter != None
  # generator, {"src": tuple(line words), "indices": i}
  # 以下两句处理了截断
  src_examples_iter = Dataset.make_examples(src_data_iter, src_seq_length_trunc, "src")
  
  # if tgt_data_iter != None:
  tgt_examples_iter = Dataset.make_examples(tgt_data_iter, tgt_seq_length_trunc, "tgt")
  # else:
  #   tgt_examples_iter = None

  # fields: {"indices":..., "src":..., "tgt":...}
  # translate, use_filter_pred = false
  dataset = Dataset(fields, src_examples_iter, tgt_examples_iter,
                        src_seq_length=src_seq_length,
                        tgt_seq_length=tgt_seq_length,
                        use_filter_pred=use_filter_pred)

  return dataset


def build_dataset_iter(datasets, fields, opt, is_train=True):
  """
  This returns user-defined train/validate data iterator for the trainer
  to iterate over. We implement simple ordered iterator strategy here,
  but more sophisticated strategy like curriculum learning is ok too.
  """
  batch_size = opt.batch_size if is_train else opt.valid_batch_size
  # batch_size_fn是自定义的取数据函数
  # if is_train and opt.batch_type == "tokens":
  #   def batch_size_fn(new, count, sofar):
  #     """
  #     In token batching scheme, the number of sequences is limited
  #     such that the total number of src/tgt tokens (including padding)
  #     in a batch <= batch_size
  #     """
  #     # Maintains the longest src and tgt length in the current batch
  #     global max_src_in_batch, max_tgt_in_batch
  #     # Reset current longest length at a new batch (count=1)
  #     if count == 1:
  #         max_src_in_batch = 0
  #         max_tgt_in_batch = 0
  #     # Src: <bos> w1 ... wN <eos>
  #     max_src_in_batch = max(max_src_in_batch, len(new.src) + 2)
  #     # Tgt: w1 ... wN <eos>
  #     max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 1)
  #     src_elements = count * max_src_in_batch
  #     tgt_elements = count * max_tgt_in_batch
  #     return max(src_elements, tgt_elements)
  # else:
  # 上面注释掉
  batch_size_fn = None

  if opt.gpu_ranks:
    device = "cuda"
  else:
    device = "cpu"

  # 载入数据时需要选择device
  return DatasetIter(datasets, fields, batch_size, batch_size_fn,
                         device, is_train)


class Dataset(torchtext.data.Dataset):
  def __init__(self, fields, src_examples_iter, tgt_examples_iter,
               src_seq_length=0, tgt_seq_length=0,
               use_filter_pred=True):

    self.src_vocabs = []
    
    def _join_dicts(*args):
      return dict(chain(*[d.items() for d in args]))

    out_fields = get_source_fields()
    # examples_iter: generator: ("src": xx, "indices": yy, "tgt": zz)
    if tgt_examples_iter is not None:
      examples_iter = (_join_dicts(src, tgt) for src, tgt in
                        zip(src_examples_iter, tgt_examples_iter))
      out_fields = get_target_fields(out_fields)
    else:
      examples_iter = src_examples_iter
      
    keys = out_fields.keys()
    # ("src", fields["src"]), ("tgt", fields["tgt"]), ("indices", fields["indices"])
    out_fields = [(k, fields[k]) for k in keys]
    # generator: 与example_iter相似，只是缺少了key且value的顺序与out_fields相同
    # generator: ((src line words), (tgt line words), index)
    example_values = ([ex[k] for k in keys] for ex in examples_iter)
    # getattr/setattr优势在于参数可以是变量
    out_examples = []
    for ex_values in example_values:
      example = torchtext.data.Example()
      for (name, field), val in zip(out_fields, ex_values):
        if field is not None:
          # preprocess没有处理什么, 相当于直接赋值
          setattr(example, name, field.preprocess(val))
        else:
          setattr(example, name, val)
      # example.src / example.tgt / example.indice
      out_examples.append(example)
    # [example, example, ....], example.indices/example.src
    def filter_pred(example):
      """ ? """
      return 0 < len(example.src) <= src_seq_length \
        and 0 < len(example.tgt) <= tgt_seq_length

    filter_pred = filter_pred if use_filter_pred else lambda x: True

    super(Dataset, self).__init__(
        out_examples, out_fields, filter_pred
    )
  def __getstate__(self):
    return self.__dict__

  def __setstate__(self, _d):
    self.__dict__.update(_d)
    
  def sort_key(self, ex):
    if hasattr(ex, "tgt"):
      return len(ex.src), len(ex.tgt)
    return len(ex.src)
    
  @staticmethod
  def make_examples(text_iter, truncate, side):
    for i, line in enumerate(text_iter):
      # 转为单词列表, split()去掉了空格
      words = line.strip().split()
      # 截断
      if truncate:
        words = words[:truncate]

      example_dict = {side: tuple(words), "indices": i}
      yield example_dict
