# -*- coding: utf-8 -*-

import torch


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    # (seq_len_src, batch_size)
    # perm是[0,1]
    # perm是[1,0]
    # x是(batch_size, seq_len_src)
    # outsize: [batch_size, seq_len_src]
    # outsize: [batch_size*count, seq_len_src]
    # batch = batch_size
    # 最终x:(seq_len_src, batch_size*count)
    # 本来是src0,src1,src2...现在是src0, src0, src0, sr...,src1,src1....
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    # x = x.permute(1,0,2)
    # 作用在数据上相当于把batch_size换到第0维

    out_size = list(x.size())
    # batch_size * count
    out_size[0] *= count
    batch = x.size(0)
    # repeat参数的1表示第一个维度(从1开始)
    # 等价于x.view(batch, -1).repeat(1,count).contiguous.view(*out_size)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, 'gpu_ranks') and len(opt.gpu_ranks) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)
