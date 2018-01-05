import sys
sys.path.append("../src/")
import data_processor as data
import constants
import models
import meter

#####################################################
# Use this for running CNN/LSTM on AskUbuntu dataset
#####################################################

#corpus = data.read_corpus(constants.corpus_path)
#embeddings, map_to_ids = data.embeddings(constants.embeddings_path)
#id_to_tensors = data.map_corpus(corpus, map_to_ids, kernel_width=1)

#train = data.get_train_data(constants.train_path, id_to_tensors)
#model = models.LSTM(300, embeddings, 0.2)
#model = models.CNN(500, 3, embeddings, 0.2)
#dev = data.get_dev_data(constants.dev_path, id_to_tensors)
#models.train_model(train, dev, model)

#####################################################
# Use this for running TF-IDF
#####################################################

#id_to_tfidf = data.featurize(constants.android_corpus_path)
#dev_annotations = data.read_annotations_android(constants.android_pos_dev_path, constants.android_neg_dev_path)
#test_annotations = data.read_annotations_android(constants.android_pos_test_path, constants.android_neg_test_path)

#print("AUC (dev):", meter.get_auc(id_to_tfidf, dev_annotations))
#print("AUC (test):", meter.get_auc(id_to_tfidf, test_annotations))

#####################################################
# Use this for running direct transfer
#####################################################
# TODO: fix src/data_processor embeddings() so that we don't have to keep changing it between 201 and 301 length

#corpus = data.read_corpus(constants.android_corpus_path)
#embeddings, map_to_ids = data.embeddings(constants.android_embeddings_path)
#id_to_tensors = data.map_corpus(corpus, map_to_ids)

#train_corpus = data.read_corpus(constants.corpus_path)
#train_id_to_tensors = data.map_corpus(train_corpus, map_to_ids)
#train = data.get_train_data(constants.train_path, train_id_to_tensors)
#dev = data.get_dev_data(constants.android_dev_path, id_to_tensors)

#model = models.CNN(700, embeddings, 0.2)
#models.train_model(train, dev, model, transfer=True)

#####################################################
# Use this for running adversarial domain adaptation
#####################################################

corpus = data.read_corpus(constants.android_corpus_path)
embeddings, map_to_ids = data.embeddings(constants.android_embeddings_path)
id_to_tensors = data.map_corpus(corpus, map_to_ids)

train_corpus = data.read_corpus(constants.corpus_path)
train_id_to_tensors = data.map_corpus(train_corpus, map_to_ids)
train = data.get_android_data(constants.train_path, train_id_to_tensors, id_to_tensors)
dev = data.get_dev_data(constants.android_dev_path, id_to_tensors)

encoder = models.CNN(500, embeddings, 0.2)
classifier = models.DomainClassifier(500, 300, 150)

models.train_adversarial_model(train, dev, encoder, classifier)
