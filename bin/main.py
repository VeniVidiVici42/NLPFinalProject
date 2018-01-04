import sys
sys.path.append("../src/")
from loss_function import loss_function
from loss_function import cs
from torch.autograd import Variable
from evaluate import Evaluation
from cnn import CNN
from meter import AUCMeter
import torch
import torch.nn as nn
import torch.nn.functional as Fvar
import numpy as np
import sklearn
import random
import time
import data_processor as data
import constants
import models

corpus = data.read_corpus(constants.corpus_path)
embeddings, map_to_ids = data.embeddings(constants.embeddings_path)
id_to_tensors = data.map_corpus(corpus, map_to_ids, kernel_width=1)

train = data.get_train_data(constants.train_path, id_to_tensors)
model = models.LSTM(300, embeddings, 0.2)
#model = models.CNN(500, 3, embeddings, 0.2)
dev = data.get_dev_data(constants.dev_path, id_to_tensors)
models.train_model(train, dev, model)

#corpus = data.read_corpus(constants.android_corpus_path)
#embeddings, map_to_ids = data.embeddings(constants.android_embeddings_path)
#id_to_tensors = data.map_corpus(corpus, map_to_ids)
#train_corpus = data.read_corpus(constants.corpus_path)
#train_id_to_tensors = data.map_corpus(train_corpus, map_to_ids)
#train = data.get_train_data(constants.train_path, train_id_to_tensors)
#model = models.LSTM(200, embeddings, 0.2)
#dev = data.get_dev_data(constants.android_dev_path, id_to_tensors)

#models.train_model(train, dev, model)

#model = torch.load("direct_transfer_model2")
#eval_path = "../data/test_android.txt"
#eval_set = dr.get_dev_data(eval_path, id_to_tensors)

#z = models.run_epoch(eval_set, False, model, None, 5)
#print(z)