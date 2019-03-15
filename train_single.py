#!/usr/bin/env python
"""
    Training on a single process
"""
import configargparse
import os
import random
import torch
import torch.nn as nn
import onmt.opts as opts

from inputters.dataset import build_dataset_iter, load_dataset, get_fields, save_fields_to_vocab, load_fields
from onmt.transformer import build_model
from utils.optimizers import build_optim
from trainer import build_trainer
from utils.logging import init_logger, logger
from collections import deque

def _check_save_model_path(opt):
  save_model_path = os.path.abspath(opt.save_model)
  model_dirname = os.path.dirname(save_model_path)
  if not os.path.exists(model_dirname):
    os.makedirs(model_dirname)

def _tally_parameters(model):
  # 模型参数总数量
  # p.nelement()
  n_params = sum([p.nelement() for p in model.parameters()])
  enc = 0
  dec = 0
  for name, param in model.named_parameters():
    if 'encoder' in name:
      enc += param.nelement()
    else:
      dec += param.nelement()
  return n_params, enc, dec

def training_opt_postprocessing(opt, device_id):
  if torch.cuda.is_available() and not opt.gpu_ranks:
    logger.info("WARNING: You have a CUDA device, \
                should run with -gpu_ranks")
  if opt.seed > 0:
    torch.manual_seed(opt.seed)
    # this one is needed for torchtext random call (shuffled iterator)
    # in multi gpu it ensures datasets are read in the same order
    random.seed(opt.seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic 作
    torch.backends.cudnn.deterministic = True
  if device_id >= 0:
    torch.cuda.set_device(device_id)
    if opt.seed > 0:
      # These ensure same initialization in multi gpu mode
      torch.cuda.manual_seed(opt.seed)
  return opt

def main(opt, device_id):
  # device_id = -1
  # 初始化gpu
  opt = training_opt_postprocessing(opt, device_id)
  init_logger(opt.log_file)
  # Load checkpoint if we resume from a previous training.
  if opt.train_from:
    logger.info('Loading checkpoint from %s' % opt.train_from)
    # Load all tensors onto the CPU
    checkpoint = torch.load(opt.train_from,
                            map_location=lambda storage, loc: storage)

    # Load default opts values then overwrite it with opts from
    # the checkpoint. It's usefull in order to re-train a model
    # after adding a new option (not set in checkpoint)
    dummy_parser = configargparse.ArgumentParser()
    opts.model_opts(dummy_parser)
    # 返回值为两个，第一个与parse_args()返回值类型相同
    default_opt = dummy_parser.parse_known_args([])[0]
    model_opt = default_opt
    # 把opt中原有的选项也加入新的参数列表中
    # 也就是说选项只可以增加而不可以删除或者修改, 
    # 如果是这样，那么后文就不需要opt了?
    model_opt.__dict__.update(checkpoint['opt'].__dict__)
  else:
    # 第一次载入
    checkpoint = None
    model_opt = opt

  # Load fields generated from preprocess phase.
  # {"src": Field, "tgt": Field, "indices": Field}
  # Field中最重要的是vocab属性，其中包含freqs、itos、stoi
  # freqs是词频，不包含特殊字符
  # src : stoi中含有<unk>、<blank>, 不含<s>与</s>
  # tgt : stoi含有<unk>、<blank>、<s>、</s>
  # <unk> = 0, <blank>(pad) = 1
  fields = load_fields(opt, checkpoint)

  # Build model.
  # 第一次应该不需要opt参数，可用model_opt代替
  model = build_model(model_opt, opt, fields, checkpoint)
  # for name, param in model.named_parameters():
  #   if param.requires_grad:
  #       print(name)
  n_params, enc, dec = _tally_parameters(model)
  logger.info('encoder: %d' % enc)
  logger.info('decoder: %d' % dec)
  logger.info('* number of parameters: %d' % n_params)
  # 没有模型保存目录则创建该目录
  _check_save_model_path(opt)

  # Build optimizer.
  optim = build_optim(model, opt, checkpoint)

  # Build model saver
  model_saver = build_model_saver(model_opt, opt, model, fields, optim)

  trainer = build_trainer(opt, device_id, model, fields,
                          optim, model_saver=model_saver)
  # 打印模型所有参数
  # for name, param in model.named_parameters():
  #   if param.requires_grad:
  #       print(param)
      
  def train_iter_fct(): 
    return build_dataset_iter(
      load_dataset("train", opt), fields, opt)

  def valid_iter_fct(): 
    return build_dataset_iter(
      load_dataset("valid", opt), fields, opt, is_train=False)

  # Do training.
  if len(opt.gpu_ranks):
    logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
  else:
    logger.info('Starting training on CPU, could be very slow')
  trainer.train(train_iter_fct, valid_iter_fct, opt.train_steps,
                opt.valid_steps)

  if opt.tensorboard:
    trainer.report_manager.tensorboard_writer.close()

def build_model_saver(model_opt, opt, model, fields, optim):
    model_saver = ModelSaver(opt.save_model,
                             model,
                             model_opt,
                             fields,
                             optim,
                             opt.save_checkpoint_steps,
                             opt.keep_checkpoint)
    return model_saver
    
class ModelSaver(object):
    """
        Base class for model saving operations
        Inherited classes must implement private methods:
            * `_save`
            * `_rm_checkpoint
    """
    def __init__(self, base_path, model, model_opt, fields, optim,
                 save_checkpoint_steps, keep_checkpoint=-1):
        # ./data/model
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.fields = fields
        self.optim = optim
        # 50
        self.keep_checkpoint = keep_checkpoint
        # 5000
        self.save_checkpoint_steps = save_checkpoint_steps

        if keep_checkpoint > 0:
            # 长度维持最大50, 初始为空
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)
    # 传入步数
    def maybe_save(self, step):
        """
        Main entry point for model saver
        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """
        if self.keep_checkpoint == 0:
            return

        if step % self.save_checkpoint_steps != 0:
            return
        # chkpt: checkoutpoint对象
        # chkpt_name: 保存的文件名
        chkpt, chkpt_name = self._save(step)

        if self.keep_checkpoint > 0:
            # 从左端pop并删除磁盘上相应文件
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            # 只添加保存模型名称
            self.checkpoint_queue.append(chkpt_name)

    def _save(self, step):
        """ Save a resumable checkpoint.

        Args:
            step (int): step number

        Returns:
            checkpoint: the saved object
            checkpoint_name: name (or path) of the saved checkpoint
        """
        # self.model
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        # self.model.generator
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)
        # 模型参数与一致缓冲区的字典, 为什么会包括generator
        model_state_dict = real_model.state_dict()
        # 排除generator
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        # generator的参数与一致缓冲区
        generator_state_dict = real_generator.state_dict()
        # 保存模型所有参数，词表，模型输入参数, 优化模型参数
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': save_fields_to_vocab(self.fields),
            'opt': self.model_opt,
            'optim': self.optim,
        }

        logger.info("Saving checkpoint %s_step_%d.pt" % (self.base_path, step))
        checkpoint_path = '%s_step_%d.pt' % (self.base_path, step)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        """
        Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """
        os.remove(name)

if __name__ == "__main__":
  parser = configargparse.ArgumentParser(
  description='train.py',
  config_file_parser_class=configargparse.YAMLConfigFileParser,
  formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
  opts.config_opts(parser)
  opts.model_opts(parser)
  opts.train_opts(parser)
  opt = parser.parse_args()
  main(opt, -1)