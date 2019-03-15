# -*- coding: utf-8 -*-
# absolute_import搞不清楚
from __future__ import absolute_import

import logging

logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
  # Child loggers propagate messages up to the handlers associated
  # with their ancestor loggers. Because of this, it is unnecessary
  # to define and configure handlers for all the loggers an application
  # uses. It is sufficient to configure handlers for a top-level logger
  # and create child loggers as needed. (You can, however,
  # turn off propagation by setting the propagate attribute of a logger
  # to False.)
  # logging.getLogger()获得的是引用，如果不存在则会创建，其不会因为函数结束而被回收
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")

  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.NOTSET)
  console_handler.setFormatter(log_format)
  # 应该使用logger.addHandler()
  logger.handlers = [console_handler]

  if log_file and log_file != '':
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_file_level)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

  return logger
