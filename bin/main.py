import sys
sys.path.append("../src/")
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