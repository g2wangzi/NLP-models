#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

# NOTE: First install bert-as-service via
# $
# $ pip install bert-serving-server
# $ pip install bert-serving-client
# $

# solving chinese law-article classification problem: https://github.com/thunlp/CAIL/blob/master/README_en.md

import json
import os
import random

#import GPUtil
import tensorflow as tf
from bert_serving.client import ConcurrentBertClient
from tensorflow.python.estimator.canned.dnn import DNNClassifier
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.training import TrainSpec, EvalSpec, train_and_evaluate

#os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUtil.getFirstAvailable()[0])
tf.logging.set_verbosity(tf.logging.INFO)

train_fp = ['./questiondata/train_5500-UTF8.label']
eval_fp = ['./questiondata/TREC_10-UTF8.label']

batch_size = 128
num_parallel_calls = 4
num_concurrent_clients = num_parallel_calls * 2  # should be at least greater than `num_parallel_calls`

bc = ConcurrentBertClient()


# file1 = open('./questiondata/train_1000.label', 'r', encoding = "ISO-8859-1") 
# Lines = file1.readlines()
# myset = set()
# for line in Lines:
#   splited = line.split(" ")
#   myset.add(splited[0])
# print(myset)

# hardcoded question-type
# question_classes = ['abb','exp','animal','body','color','cremat','currency',
#                     'dismed','event','food','instru','lang','letter','other',
#                     'plant','product','religion','sport','substance','symbol','techmeth',
#                     'termeq','veh','word','def','manner','reason','gr',
#                     'ind','title','desc','city','country','mount','other','state',
#                     'code','count','date','dist','money','order','other','period','percent',
#                     'speed','temp','size','weight']

question_classes = ['ENTY:other', 'NUM:dist', 'HUM:gr', 'NUM:speed', 'ENTY:instru', 'ENTY:product', 'ENTY:letter', 
                  'ENTY:veh', 'ENTY:color', 'LOC:other', 'NUM:period', 'ENTY:event', 'ENTY:sport', 'ENTY:animal', 'LOC:country', 
                  'ENTY:symbol', 'NUM:count', 'HUM:ind', 'DESC:reason', 'ENTY:plant', 'HUM:desc', 'DESC:def', 'ENTY:religion', 
                  'ABBR:exp', 'ENTY:body', 'ENTY:word', 'ENTY:termeq', 'ENTY:lang', 'NUM:perc', 'ENTY:cremat', 'LOC:state', 
                  'ENTY:techmeth', 'ENTY:food', 'NUM:ord', 'NUM:volsize', 'HUM:title', 'NUM:date', 'LOC:city', 'DESC:manner', 
                  'ENTY:dismed', 'LOC:mount', 'NUM:money', 'ENTY:substance', 'NUM:other', 'NUM:code', 'ENTY:currency', 'NUM:temp', 
                  'ABBR:abb', 'NUM:weight', 'DESC:desc']


def get_encodes(x):

    text = [line.decode("utf-8").split(" ", 1)[1] for line in x]
    features = bc.encode(text)
    labels = [[line.decode("utf-8").split(" ", 1)[0]] for line in x]

    return features, labels


config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
run_config = RunConfig(model_dir='./data/cips/save/question-all-model-v2',
                       session_config=config,
                       save_checkpoints_steps=500)

estimator = DNNClassifier(
    hidden_units=[512],
    feature_columns=[tf.feature_column.numeric_column('feature', shape=(1024,))],   
    n_classes=len(question_classes),
    config=run_config,
    label_vocabulary=question_classes,
    dropout=0.1)

input_fn = lambda fp: (tf.data.TextLineDataset(fp)
                       .apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
                       .batch(batch_size)
                       .map(lambda x: tf.py_func(get_encodes, [x], [tf.float32, tf.string], name='bert_client'),
                            num_parallel_calls=num_parallel_calls)
                       .map(lambda x, y: ({'feature': x}, y))
                       .prefetch(20))

train_spec = TrainSpec(input_fn=lambda: input_fn(train_fp), max_steps=3000)


eval_spec = EvalSpec(input_fn=lambda: input_fn(eval_fp), throttle_secs=0)

train_and_evaluate(estimator, train_spec, eval_spec)