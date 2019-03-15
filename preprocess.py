# 命令行参数控制
import configargparse
# 根据类似unix shell的正则表达式规则匹配
import glob
import os
# 
import codecs
import gc

import torch
import torchtext.vocab
from collections import Counter, OrderedDict

import onmt.constants as Constants
import onmt.opts as opts
from inputters.dataset import get_fields, build_dataset, make_text_iterator_from_file
from utils.logging import init_logger, logger


def save_fields_to_vocab(fields):
  """
  Save Vocab objects in Field objects to `vocab.pt` file.
  """
  vocab = []
  for k, f in fields.items():
    if f is not None and 'vocab' in f.__dict__:
      # 这句话在干嘛？
      f.vocab.stoi = f.vocab.stoi
      vocab.append((k, f.vocab))
  return vocab

def build_field_vocab(field, counter, **kwargs):
  # OrderedDict按key插入顺序排序字典
  # fromkeys的参数是可迭代对象, 此处value为None

  # specials = ['<unk>', '<blank>', '<s>', '</s>']
  # src与tgt的差别就在这里，if tok is not None 排除了 <s> , </s>
  # 把填充字符换为field.pad_token
  specials = list(OrderedDict.fromkeys(
    tok for tok in [field.pad_token, field.unk_token, field.init_token,
                    field.eos_token]
    if tok is not None))
  # vocab属性是自己加上的, vocab里面含有itos:list与stoi:dict
  field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)

def merge_vocabs(vocabs, vocab_size=None, min_frequency=1):
  merged = sum([vocab.freqs for vocab in vocabs], Counter())
  return torchtext.vocab.Vocab(merged,
                               specials=[Constants.UNK_WORD, Constants.PAD_WORD,
                                         Constants.BOS_WORD, Constants.EOS_WORD],
                               max_size=vocab_size,
                               min_freq=min_frequency)    

def build_vocab(train_dataset_files, fields, share_vocab,
                src_vocab_size, src_words_min_frequency,
                tgt_vocab_size, tgt_words_min_frequency):
  counter = {}

  # Counter为dict的子类
  # key: Counter(), 统计"src"、"tgt"单词出现次数
  for k in fields:
    counter[k] = Counter()

  # Load vocabulary
  for _, path in enumerate(train_dataset_files):
    dataset = torch.load(path)
    logger.info(" * reloading %s." % path)
    # [example example example ... ]
    for ex in dataset.examples:
      for k in fields:
        val = getattr(ex, k, None)
        # 如果是indics
        if not fields[k].sequential:
          continue
        # update的参数是可迭代对象,此处是元组
        # 这样遍历一遍就可以统计好词表
        counter[k].update(val)

    dataset.examples = None
    gc.collect()
    del dataset.examples
    gc.collect()
    del dataset
    gc.collect()

  # tgt: freq是词出现的频率，不包含特殊字符
  # itos是index到word的映射，频率从高到低，最前面是4个特殊字符
  # 0 : unk, 1 : blank, 2 : s, 3 : /s
  # 注意顺序乱掉了
  # stoi与itos相对应
  build_field_vocab(fields["tgt"], counter["tgt"],
                     max_size=tgt_vocab_size,
                     min_freq=tgt_words_min_frequency)
  logger.info(" * tgt vocab size: %d." % len(fields["tgt"].vocab))

  # 为什么这里只出现<unk>与<blank> 原因是field.init_token
  # field.eos_token不存在
  build_field_vocab(fields["src"], counter["src"],
                     max_size=src_vocab_size,
                     min_freq=src_words_min_frequency)
  logger.info(" * src vocab size: %d." % len(fields["src"].vocab))

  # Merge the input and output vocabularies.
  if share_vocab:
    # `tgt_vocab_size` is ignored when sharing vocabularies
    logger.info(" * merging src and tgt vocab...")
    merged_vocab = merge_vocabs(
        [fields["src"].vocab, fields["tgt"].vocab],
        vocab_size=src_vocab_size,
        min_frequency=src_words_min_frequency)
    fields["src"].vocab = merged_vocab
    fields["tgt"].vocab = merged_vocab

  return fields

def parse_args():
  parser = configargparse.ArgumentParser(
    description='preprocess.py',
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

  # 预处理所需要的全部参数均在opts模块中
  opts.config_opts(parser)
  opts.preprocess_opts(parser)

  opt = parser.parse_args()
  # 指定一个随机数会使得每次按顺序使用rand...产生的数值均相同
  # 不知道在不同机器上是否相同, 如果相同的话结果便可以复现
  torch.manual_seed(opt.seed)

  return opt

def build_save_in_shards_using_shards_size(src_corpus, tgt_corpus, fields,
                                           corpus_type, opt):
  src_data = []
  tgt_data = []
  with open(src_corpus, "r") as src_file:
    with open(tgt_corpus, "r") as tgt_file:
      for s, t in zip(src_file, tgt_file):
        src_data.append(s)
        tgt_data.append(t)
  if len(src_data) != len(tgt_data):
    raise AssertionError("Source and Target should \
                           have the same length")

  # why not //?
  num_shards = int(len(src_data) / opt.shard_size)
  for x in range(num_shards):
    logger.info("Splitting shard %d." % x)
    f = codecs.open(src_corpus + ".{0}.txt".format(x), "w",
                    encoding="utf-8")
    f.writelines(
            src_data[x * opt.shard_size: (x + 1) * opt.shard_size])
    f.close()
    f = codecs.open(tgt_corpus + ".{0}.txt".format(x), "w",
                    encoding="utf-8")
    f.writelines(
            tgt_data[x * opt.shard_size: (x + 1) * opt.shard_size])
    f.close()
  num_written = num_shards * opt.shard_size
  if len(src_data) > num_written:
    logger.info("Splitting shard %d." % num_shards)
    f = codecs.open(src_corpus + ".{0}.txt".format(num_shards),
                    'w', encoding="utf-8")
    f.writelines(
            src_data[num_shards * opt.shard_size:])
    f.close()
    f = codecs.open(tgt_corpus + ".{0}.txt".format(num_shards),
                    'w', encoding="utf-8")
    f.writelines(
            tgt_data[num_shards * opt.shard_size:])
    f.close()
  src_list = sorted(glob.glob(src_corpus + '.*.txt'))
  tgt_list = sorted(glob.glob(tgt_corpus + '.*.txt'))

  ret_list = []

  for index, src in enumerate(src_list):
    logger.info("Building shard %d." % index)
    src_iter = make_text_iterator_from_file(src)
    tgt_iter = make_text_iterator_from_file(tgt_list[index])
    dataset = build_dataset(
      fields,
      src_iter,
      tgt_iter,
      src_seq_length=opt.src_seq_length,
      tgt_seq_length=opt.tgt_seq_length,
      src_seq_length_trunc=opt.src_seq_length_trunc,
      tgt_seq_length_trunc=opt.tgt_seq_length_trunc
    )

    pt_file = "{:s}_{:s}.{:d}.pt".format(
      opt.save_data, corpus_type, index)

    # We save fields in vocab.pt seperately, so make it empty.
    dataset.fields = []

    logger.info(" * saving %sth %s data shard to %s."
                % (index, corpus_type, pt_file))
    torch.save(dataset, pt_file)

    ret_list.append(pt_file)
    os.remove(src)
    os.remove(tgt_list[index])
    del dataset.examples
    gc.collect()
    del dataset
    gc.collect()

  return ret_list

def build_save_vocab(train_dataset, fields, opt):
  """ Building and saving the vocab """
  fields = build_vocab(train_dataset, fields,
                                 opt.share_vocab,
                                 opt.src_vocab_size,
                                 opt.src_words_min_frequency,
                                 opt.tgt_vocab_size,
                                 opt.tgt_words_min_frequency)

  # Can't save fields, so remove/reconstruct at training time.
  vocab_file = opt.save_data + '_vocab.pt'
  torch.save(save_fields_to_vocab(fields), vocab_file)
    
def build_save_dataset(corpus_type, fields, opt):
  """ Building and saving the dataset """
  assert corpus_type in ['train', 'valid']

  if corpus_type == 'train':
    src_corpus = opt.train_src
    tgt_corpus = opt.train_tgt
  else:
    src_corpus = opt.valid_src
    tgt_corpus = opt.valid_tgt

  # if (opt.shard_size > 0):
  #   return build_save_in_shards_using_shards_size(src_corpus,
  #                                                 tgt_corpus,
  #                                                 fields,
  #                                                 corpus_type,
  #                                                 opt)

  # We only build a monolithic dataset.
  # But since the interfaces are uniform, it would be not hard
  # to do this should users need this feature.
  # 每次产生一行
  src_iter = make_text_iterator_from_file(src_corpus)
  tgt_iter = make_text_iterator_from_file(tgt_corpus)
  dataset = build_dataset(
    fields,
    src_iter,
    tgt_iter,
    src_seq_length=opt.src_seq_length,
    tgt_seq_length=opt.tgt_seq_length,
    src_seq_length_trunc=opt.src_seq_length_trunc,
    tgt_seq_length_trunc=opt.tgt_seq_length_trunc)

  # We save fields in vocab.pt seperately, so make it empty.
  # 仅保留self.example.src / self.example.tgt / self.example.indice
  dataset.fields = []

  pt_file = "{:s}_{:s}.pt".format(opt.save_data, corpus_type)
  logger.info(" * saving %s dataset to %s." % (corpus_type, pt_file))
  torch.save(dataset, pt_file)

  return [pt_file]

def main():
  opt = parse_args()

  if (opt.shuffle > 0):
    raise AssertionError("-shuffle is not implemented, please make sure \
                         you shuffle your data before pre-processing.")
  print (opt)
  # 全部日志写入file以及console
  init_logger(opt.log_file)
  logger.info("Extracting features...")

  logger.info("Building `Fields` object...")
  fields = get_fields()

  logger.info("Building & saving training data...")
  train_dataset_files = build_save_dataset('train', fields, opt)

  logger.info("Building & saving validation data...")
  build_save_dataset('valid', fields, opt)

  logger.info("Building & saving vocabulary...")

  build_save_vocab(train_dataset_files, fields, opt)

if __name__ == "__main__":
  main()
  