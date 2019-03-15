"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

from utils.loss import build_loss_compute
from utils.logging import logger
from utils.report_manager import build_report_manager
from utils.statistics import Statistics
from utils.distributed import all_gather_list, all_reduce_and_rescale_tensors
from inputters.dataset import make_features

def build_trainer(opt, device_id, model, fields,
                  optim, model_saver=None):
  """
  Simplify `Trainer` creation based on user `opt`s*

  Args:
      opt (:obj:`Namespace`): user options (usually from argument parsing)
      model (:obj:`onmt.models.NMTModel`): the model to train
      fields (dict): dict of fields
      optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
      model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
          used to save the model
  """
  # 传词表是干嘛的？
  train_loss = build_loss_compute(
    model, fields["tgt"].vocab, opt)
  valid_loss = build_loss_compute(
    model, fields["tgt"].vocab, opt, train=False)

  # 截断后向传播？默认是0
  trunc_size = opt.truncated_decoder  # Badly named...
  # 默认是2, 可以并行运行generator
  shard_size = opt.max_generator_batches
  # 梯度的normalization方法: tokens
  norm_method = opt.normalization
  # 一次更新梯度，batch_size * accum_count, 默认accum_count = 1 
  grad_accum_count = opt.accum_count
  # 这里gpu的数量指定为分布式进程的数量
  n_gpu = opt.world_size
  if device_id >= 0:
    gpu_rank = opt.gpu_ranks[device_id]
  else:
    gpu_rank = 0
    n_gpu = 0
  # 啥玩意
  gpu_verbose_level = opt.gpu_verbose_level

  report_manager = build_report_manager(opt)
  trainer = Trainer(model, train_loss, valid_loss, optim, trunc_size,
                         shard_size, norm_method,
                         grad_accum_count, n_gpu, gpu_rank,
                         gpu_verbose_level, report_manager,
                         model_saver=model_saver)
  return trainer


class Trainer(object):
  """
  Class that controls the training process.

  Args:
      model(:py:class:`onmt.models.model.NMTModel`): translation model
          to train
      train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
         training loss computation
      valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
         training loss computation
      optim(:obj:`onmt.utils.optimizers.Optimizer`):
         the optimizer responsible for update
      trunc_size(int): length of truncated back propagation through time
      shard_size(int): compute loss in shards of this size for efficiency
      norm_method(string): normalization methods: [sents|tokens]
      grad_accum_count(int): accumulate gradients this many times.
      report_manager(:obj:`onmt.utils.ReportMgrBase`):
          the object that creates reports, or None
      model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
          used to save a checkpoint.
          Thus nothing will be saved if this parameter is None
  """

  def __init__(self, model, train_loss, valid_loss, optim,
               trunc_size=0, shard_size=32,
               norm_method="sents", grad_accum_count=1, n_gpu=1, gpu_rank=1,
               gpu_verbose_level=0, report_manager=None, model_saver=None):
    # Basic attributes.
    self.model = model
    self.train_loss = train_loss
    self.valid_loss = valid_loss
    self.optim = optim
    self.trunc_size = trunc_size
    self.shard_size = shard_size
    self.norm_method = norm_method
    self.grad_accum_count = grad_accum_count
    self.n_gpu = n_gpu
    self.gpu_rank = gpu_rank
    self.gpu_verbose_level = gpu_verbose_level
    self.report_manager = report_manager
    self.model_saver = model_saver

    assert grad_accum_count > 0
    if grad_accum_count > 1:
      assert(self.trunc_size == 0), \
        """To enable accumulated gradients,
           you must disable target sequence truncating."""

    # Set model in training mode.
    self.model.train()

  def train(self, train_iter_fct, valid_iter_fct, train_steps, valid_steps):
    """
    The main training loops.
    by iterating over training data (i.e. `train_iter_fct`)
    and running validation (i.e. iterating over `valid_iter_fct`

    Args:
        train_iter_fct(function): a function that returns the train
            iterator. e.g. something like
            train_iter_fct = lambda: generator(*args, **kwargs)
        valid_iter_fct(function): same as train_iter_fct, for valid data
        train_steps(int):
        valid_steps(int):
        save_checkpoint_steps(int):

    Return:
        None
    """
    logger.info('Start training...')

    # step = 1, 优化步数+1
    step = self.optim._step + 1
    normalization = 0
    # 训练数据迭代器
    train_iter = train_iter_fct()

    # 从这里开始计时
    total_stats = Statistics()
    report_stats = Statistics()
    # report_manager初始化设置为total_stats的时间
    self._start_report_manager(start_time=total_stats.start_time)

    # train_steps = 150000
    while step <= train_steps:
      reduce_counter = 0
      # 1. batch第二次取得时候是否为空, 不为空值
      # 2. 在field.process()中使用了转置，对于src而言是(xx*batch_size, length)
      # 对tgt而言也使用了转置, xx * batch_size, 对于 indices 是 batch_size

      # src 每一句话的末尾都是<s>, tgt每一句话开头为<s>， 末尾是</s>, 填充为<blank> = 1
      # 这个下次再看为什么
      for i, batch in enumerate(train_iter):
        if self.norm_method == "tokens":
          # 去掉init_token不统计, 不去掉eos_token
          # 非填充字符数量
          num_tokens = batch.tgt[1:].ne(
            self.train_loss.padding_idx).sum()
          # 统计的字符数量包括</s>
          normalization += num_tokens.item()
        else:
          normalization += batch.batch_size
        reduce_counter += 1
        # 梯度累加
        self._gradient_accumulation(
          batch, normalization, total_stats,
          report_stats)
        report_stats = self._maybe_report_training(
          step, train_steps,
          self.optim.learning_rate,
          report_stats)

        normalization = 0
        if (step % valid_steps == 0):
          if self.gpu_verbose_level > 0:
            logger.info('GpuRank %d: validate step %d'
                          % (self.gpu_rank, step))
          valid_iter = valid_iter_fct()
          valid_stats = self.validate(valid_iter)
          if self.gpu_verbose_level > 0:
            logger.info('GpuRank %d: gather valid stat \
                          step %d' % (self.gpu_rank, step))
          valid_stats = self._maybe_gather_stats(valid_stats)
          if self.gpu_verbose_level > 0:
            logger.info('GpuRank %d: report stat step %d'
                          % (self.gpu_rank, step))
          self._report_step(self.optim.learning_rate,
                            step, valid_stats=valid_stats)

        if self.gpu_rank == 0:
          self._maybe_save(step)
        step += 1
        if step > train_steps:
          break
      if self.gpu_verbose_level > 0:
        logger.info('GpuRank %d: we completed an epoch \
                    at step %d' % (self.gpu_rank, step))
      train_iter = train_iter_fct()

    return total_stats

  def validate(self, valid_iter):
    """ Validate model.
        valid_iter: validate data iterator
    Returns:
        :obj:`nmt.Statistics`: validation loss statistics
    """
    # Set model in validating mode.
    self.model.eval()

    stats = Statistics()

    for batch in valid_iter:
      src = make_features(batch, 'src')
      _, src_lengths = batch.src

      tgt = make_features(batch, 'tgt')

      # F-prop through the model.
      logits = self.model(src, tgt, src_lengths)

      # Compute loss.
      batch_stats = self.valid_loss.monolithic_compute_loss(
        batch, logits, None)

      # Update statistics.
      stats.update(batch_stats)

    # Set model back to training mode.
    self.model.train()
    return stats

  def _gradient_accumulation(self, batch, normalization, total_stats,
                             report_stats):
      # 1. src = batch.src[0],  xx * batch_size, 最后统一以<s>结尾？
      src = make_features(batch, 'src')
      # 2. src_lengths = batch.src[1], batch_size
      _, src_lengths = batch.src
      # 3. tgt_outer = batch.tgt, yy * batch_size, 包括开头的<s>2与结尾的</s>3以及可能出现的填充字符<blank>1
      tgt = make_features(batch, 'tgt')

      # 目标句子长度
      target_size = tgt.size(0)

      # 2. F-prop all but generator.
      # batch之间梯度无需累加
      self.model.zero_grad()
      # outputs: (len, batch, dim)
      # attns: (len_tgt, batch, len_src)
      # logits: (len, batch_size, 2048)
      logits = self.model(src, tgt, src_lengths)

      # 3. Compute loss in shards for memory efficiency.
      # self.shard_size默认是2, attns没用上？
      batch_stats = self.train_loss.sharded_compute_loss(
          batch, logits, None, 0,
          target_size, self.shard_size, normalization)

      total_stats.update(batch_stats)
      report_stats.update(batch_stats)

      # 4. Update the parameters and statistics.
      self.optim.step()

      # If truncated, don't backprop fully.
      # TO CHECK
      # if dec_state is not None:
      #    dec_state.detach()
      if self.model.decoder.state is not None:
          self.model.decoder.detach_state()

  def _start_report_manager(self, start_time=None):
      """
      Simple function to start report manager (if any)
      """
      if self.report_manager is not None:
          if start_time is None:
              self.report_manager.start()
          else:
              self.report_manager.start_time = start_time

  def _maybe_gather_stats(self, stat):
      """
      Gather statistics in multi-processes cases

      Args:
          stat(:obj:onmt.utils.Statistics): a Statistics object to gather
              or None (it returns None in this case)

      Returns:
          stat: the updated (or unchanged) stat object
      """
      if stat is not None and self.n_gpu > 1:
          return Statistics.all_gather_stats(stat)
      return stat

  def _maybe_report_training(self, step, num_steps, learning_rate,
                             report_stats):
      """
      Simple function to report training stats (if report_manager is set)
      see `onmt.utils.ReportManagerBase.report_training` for doc
      """
      if self.report_manager is not None:
          return self.report_manager.report_training(
              step, num_steps, learning_rate, report_stats,
              multigpu=self.n_gpu > 1)

  def _report_step(self, learning_rate, step, train_stats=None,
                   valid_stats=None):
      """
      Simple function to report stats (if report_manager is set)
      see `onmt.utils.ReportManagerBase.report_step` for doc
      """
      if self.report_manager is not None:
          return self.report_manager.report_step(
              learning_rate, step, train_stats=train_stats,
              valid_stats=valid_stats)

  def _maybe_save(self, step):
      """
      Save the model if a model saver is set
      """
      if self.model_saver is not None:
          self.model_saver.maybe_save(step)
