"""
This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
import onmt.constants as Constants
from utils.misc import use_gpu
from utils.statistics import Statistics


def build_loss_compute(model, tgt_vocab, opt, train=True):
  """
  Returns a LossCompute subclass which wraps around an nn.Module subclass
  (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
  object allows this loss to be computed in shards and passes the relevant
  data to a Statistics object which handles training/validation logging.
  Currently, the NMTLossCompute class handles all loss computation except
  for when using a copy mechanism. Despite their name, LossCompute objects
  do not merely compute the loss but also perform the backward pass inside
  their sharded_compute_loss method.
  """
  device = torch.device("cuda" if use_gpu(opt) else "cpu")

  # 1
  padding_idx = tgt_vocab.stoi[Constants.PAD_WORD]
  # label_smoothing = 0.1
  if opt.label_smoothing > 0 and train:
    criterion = LabelSmoothingLoss(
        opt.label_smoothing, len(tgt_vocab), ignore_index=padding_idx
    )
  else:
      # 输入必须是经过log_softmax之后的数值
    criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')

    # 对于criterion:
    # input (FloatTensor): batch_size x n_classes 
    # output (LongTensor): batch_size
    # 对于loss_gen
    # i

  # if the loss function operates on vectors of raw logits instead of
  # probabilities, only the first part of the generator needs to be
  # passed to the NMTLossCompute. At the moment, the only supported
  # loss function of this kind is the sparsemax loss.
  # 获得概率分布 ==> (dim, len(vocab)) + softmax(-1) 
  # 共用model.generator的权重
  loss_gen = model.generator
  compute = NMTLossCompute(criterion, loss_gen)
  # 网络参数送入cpu/gpu中计算
  compute.to(device)

  return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating multiple
    loss computations

    实现_compute_loss(), _make_shard_state()
    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, criterion, generator):
        super(LossComputeBase, self).__init__()
        self.criterion = criterion
        self.generator = generator

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    # outputs: (len, batch, dim)
    # attns: (len_tgt, batch, len_src)
    # cur_trunc: 0
    # trunc_size : target_size
    # shard_size: 2, Maximum batches of words in a sequence to run
    # the generator on in parallel. Higher is faster, but
    # uses more memory.
    # normalization: tgt中除去<s>, 不等于填充字符的数量
    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        batch_stats = Statistics()
        # 0-len_tgt_size 
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns)
        '''
            return {
                "output": output,
                "target": batch.tgt[1:len_tgt_size]
            }
        '''
        # shard_size被设置为2
        # shard: {"output": output_, "target": target_}
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            # backward, 这里的div?
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets
        1 / x * vocab_size / x
        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """ 
        # 返回行的最大值的索引即预测的单词
        # (shard_size*batch)
        pred = scores.max(1)[1]
        # (shard_size*batch)
        non_padding = target.ne(self.padding_idx)
        # 排除掉填充字符，得到正确的单词个数
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        # 总的单词个数
        num_non_padding = non_padding.sum().item()
        return Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        # 值为1 <unk>, 填充字符
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()
        # 0.1
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        # one_hot: tgt_vocab_size
        # 一维向量，长度是tgt_vocab_size, 数值全部为smoothing_value接近于0
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        # 填充字符不参与计算
        one_hot[self.ignore_index] = 0
        # register_buffer什么时候用到?
        # 1 * len(vocab_size_tgt)
        # unsqueeze是共享内存的, 模型无需更新状态
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """

        # model_prob: t_batch_size * vocab_size
        model_prob = self.one_hot.repeat(target.size(0), 1)
        # target.unsqueeze(1): t_batch_size * 1
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, criterion, generator, normalization="sents"):
        super(NMTLossCompute, self).__init__(criterion, generator)

    def _make_shard_state(self, batch, output, range_, attns=None):
        # output: len * batch_size * dim
        # batch.tgt: len * batch_size
        # target, 排除掉<s>
        '''
            return {
                "output": output,
                "target": batch.tgt[1:len_tgt_size]
            }
        '''
        return {
            "output": output,
            # "target": batch.tgt[range_[0] + 1: range_[1]],
            "target": batch.tgt[range_[0]: range_[1]],
        }

    def _compute_loss(self, batch, output, target):
        # output: output_(shard_size, len_tgt, dim)
        # target: target_(shard_size, len_tgt)
        # bottled_output: (shard_size, len_tgt, dim)
        bottled_output = self._bottle(output)
        # 不管这么变，最终处理的是dim，因而不会产生影响
        # generator: 1. =>vocab, 2.softmax 
        # scores: (shard_Size*batch, vocab)
        scores = self.generator(bottled_output)
        # target: (shard_size*batch)
        gtruth = target.view(-1)
        # x * vocab_size / x
        loss = self.criterion(scores, gtruth)
        # loss : 
        # loss的梯度没有复制, 这里是为了统计数据
        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats


def filter_shard_state(state, shard_size=None):
    '''
        state: {
            "output": output,
            "target": batch.tgt[1:len_tgt_size]
        }
        shard_size: 2
    '''
    for k, v in state.items():
        if shard_size is None:
            yield k, v
        # shard_size针对的是len, 而不是batch_size
        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                # output分为两份, 在第一个维度上划分, a >= b
                for v_chunk in torch.split(v, shard_size):
                    # torch.split产生的不是新的tensor, clone并没有克隆梯度 
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    '''
        state: {
            "output": output,
            "target": batch.tgt[1:len_tgt_size]
        }
        shard_size: 2
    '''
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        # {"output": (output总， 若干output分), "target": (target总, 若干target分)}
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        # keys: ("output", "target")
        # values: (若干个output_列表, 若干个target_列表)
        # output_: shard_size * batch_size * dim
        # target_: shard_size * batch_size
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.

        # {"output": output_, "target": target_}
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
