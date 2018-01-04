import sys
import gzip
import random
import torch
import numpy as np
import constants
from sklearn.feature_extraction.text import TfidfVectorizer

'''
Source: https://github.com/taolei87/rcnn/blob/master/code/qa/myio.py
'''
def read_corpus(path):
	empty_cnt = 0
	raw_corpus = {}
	fopen = gzip.open if path.endswith(".gz") else open
	with fopen(path) as fin:
		for line in fin:
			try:
				id, title, body = line.decode("utf-8").split("\t")
			except:
				id, title, body = line.split("\t")
			title, body = title.lower(), body.lower()
			if len(title) == 0:
				empty_cnt += 1
				continue
			title = title.strip().split()
			body = body.strip().split()
			raw_corpus[id] = (title, body)
	return raw_corpus
	
'''
Source: https://github.com/taolei87/rcnn/blob/master/code/qa/myio.py
'''
def read_annotations(path, K_neg=20, prune_pos_cnt=10):
	lst = [ ]
	with open(path) as fin:
		for line in fin:
			parts = line.split("\t")
			pid, pos, neg = parts[:3]
			pos = pos.split()
			neg = neg.split()
			if len(pos) == 0 or (len(pos) > prune_pos_cnt and prune_pos_cnt != -1): continue
			if K_neg != -1:
				random.shuffle(neg)
				neg = neg[:K_neg + 2] # decreases chance of bad luck
			s = set()
			qids = [ ]
			qlabels = [ ]
			for q in neg:
				if q not in s:
					qids.append(q)
					qlabels.append(0 if q not in pos else 1)
					s.add(q)
			for q in pos:
				if q not in s:
					qids.append(q)
					qlabels.append(1)
					s.add(q)
			lst.append((pid, qids, qlabels))

	return lst
	
	
'''
Processes pretrained word embeddings.

Returns an array of word embeddings as well as a map of words to indices in the array. Each embedding has length exactly 201. The first 200 coordinates correspond to the given pretrained embeddings, while the 201st coordinate is an indicator for whether the word is known.

The embedding array has two special embeddings prepended. The first entry is an identically zero vector to be used for padding purposes. The second entry is also an identically zero vector except that the 201st coordinate is 1. This is used for unknown words. All other entries are pretrained embeddings of some word.

@param path The path to the file containing the pretrained word embeddings.
@return A tuple (embeddings, word_to_idx) consisting of an augmented array of embeddings as described above, and a dictionary mapping words to their index in the list of embeddings.
'''
def embeddings(path):
	embeddings = []
	word_to_idx = {}
	# Create padding tensor
	embeddings.append(np.zeros(301)) 
	# Create unknown word tensor
	unk_embedding = np.zeros(301) 
	unk_embedding[-1] = 1 
	embeddings.append(unk_embedding)
	# Process remaining embeddings
	fopen = gzip.open if path.endswith(".gz") else open
	idx = 0
	with fopen(path) as f:
		for line in f:
			word, vector = line.split()[0], line.split()[1:]
			vector = [float(x) for x in vector]
			vector.append(0)
			if len(embeddings) is 0 or len(vector) == len(embeddings[0]):
				embeddings.append(vector)
				# Words might be in unicode string format or byte string format. In latter case need to transform to match corpus processing.
				try:
					word_to_idx[word.decode('utf-8')] = idx + 2
				except:
					word_to_idx[word] = idx + 2
				idx = idx + 1
	return np.array(embeddings), word_to_idx

'''
Transforms a corpus of questions into vectors of the word's indices in a list of embeddings.

The provided corpus consists of question objects of the form (id, title, body). The title and body are both transformed into index vectors according to the provided map, which should presumably be sourced from embeddings(). Because these index vectors are padded, this additionally constructs masks indicating which indices of the index vectors are non-padding, normalized so that the sum of the mask's coordinates is 1. This then returns a dictionary of ids to index vectors and masks.

@param corpus The corpus to process. This is a dictionary mapping ids to (title, body) pairs
@param word_to_idx A mapping of words to indices, presumably sourced from embeddings() and thus pointing to indices in a list of embeddings (provided earlier)
@param kernel_width If using a CNN, the width of the kernel to use. This is necessary as the mask needs to be appropriately modified.
@return A dictionary mapping ids to 4-tuples (title index vector, title mask, body index vector, body mask)
'''
def map_corpus(corpus, word_to_idx, kernel_width = 3):
	id_to_tensors = {}
	for id in corpus:
		(title, body) = corpus[id]
		title_tensor, title_mask = index_tensor(title, word_to_idx, constants.title_length, kernel_width = kernel_width)
		body_tensor, body_mask = index_tensor(body, word_to_idx, constants.body_length, kernel_width = kernel_width) 
		id_to_tensors[id] = (title_tensor, title_mask, body_tensor, body_mask)
	return id_to_tensors

	
'''
Transforms text to a vector of indices, pads the vector if necessary, and includes the normalized mask indicating which indices in the vector are not padding.

@param text The text to transform
@param word_to_idx A mapping of words to indices, presumably sourced from embeddings() and thus pointing to indices in a list of embeddings (provided earlier)
@param max_length The maximum length of the vector. If the text is longer than this length, consider only the first (length) words. If the text is shorter than this length, the resulting vectors will be padded to this length.
@param kernel_width If using a CNN, the width of the kernel to use. This is necessary as the mask needs to be appropriately modified.
'''
def index_tensor(text, word_to_idx, max_length, kernel_width = 3):
	# Build tensor, replacing unknown words with 1 (recall this maps to unknown word in embedding list)
	tensor = [word_to_idx[word] if word in word_to_idx else 1 for word in text][:max_length] # Truncate at max_length if necessary
	mask = []
	if len(tensor) < max_length:
		# Pad with zeros
		tensor.extend([0 for i in range(max_length - len(tensor))])
	# Add mask for non-padding words
	mask.extend([1.0/(len(text) - kernel_width + 1) for i in range(len(text) - kernel_width + 1)][:max_length - kernel_width + 1])
	# Add mask for padding words
	mask.extend([0 for i in range(max_length - kernel_width + 1 - len(mask))])
	return torch.LongTensor(tensor), torch.FloatTensor(mask)

'''
Processes training data. This is given in the format (question id, similar question ids, random question ids). This transforms the data into instances of:
	(title, body, candidate titles, candidate bodies)
where the candidates consist of a single similar question and all the random questions (restricted to 20). Each instance also includes the associated masks.

@param path The path to read the training data from
@param id_to_tensors A map from question ids to text embeddings (sourced from map_corpus())
@return A list of training data in dictionary form. Each instance is a dictionary with the following fields: pid_title, pid_title_mask, pid_body, pid_body_mask, candidate_titles, candidate_titles_mask, candidate_body, candidate_body_mask. The candidate titles in turn consist of the index tensors for each candidate question (one positive question followed by 20 random questions) concatenated and similarly for the candidate bodies.
'''
def get_train_data(path, id_to_tensors):
	train_data = []
	data = read_annotations(path)
	for (pid, qid, qlabels) in data:
		if pid not in id_to_tensors:
			continue
		sim_id = [] # similar questions
		rand_id = [] # random questions
		for i in range(len(qid)):
			if qid[i] not in id_to_tensors:
				continue
			if qlabels[i] is 1:
				sim_id.append(qid[i])
			else:
				rand_id.append(qid[i])
		rand_id = rand_id[:20]
		for id in sim_id:
			(pid_title, pid_title_mask, pid_body, pid_body_mask) = id_to_tensors[pid]
			candidate_titles = torch.cat([torch.unsqueeze(id_to_tensors[x][0],0) for x in [id] + rand_id])
			candidate_titles_mask = torch.cat([torch.unsqueeze(id_to_tensors[x][1],0) for x in [id] + rand_id])
			candidate_body = torch.cat([torch.unsqueeze(id_to_tensors[x][2],0) for x in [id] + rand_id])
			candidate_body_mask = torch.cat([torch.unsqueeze(id_to_tensors[x][3],0) for x in [id] + rand_id])
			train_data.append({
								'pid_title': pid_title,
								'pid_title_mask': pid_title_mask,
								'pid_body': pid_body,
								'pid_body_mask': pid_body_mask,
								'candidate_titles': candidate_titles,
								'candidate_titles_mask': candidate_titles_mask,
								'candidate_body': candidate_body,
								"candidate_body_mask": candidate_body_mask
								})
	return train_data

'''
Processes testing data. This is given in the format (question id, similar question ids, random question ids). This transforms the data into instances of:
	(title, body, candidate titles, candidate bodies, labels)
where the candidates consist of a single similar question and all the random questions. Each instance also includes the associated masks. Note there are slight differences between this and get_train_data(): this function includes labels of questions, and the number of random questions is not restricted here (as we are evaluating off this set, not training).

@param path The path to read the training data from
@param id_to_tensors A map from question ids to text embeddings (sourced from map_corpus())
@return A list of training data in dictionary form. Each instance is a dictionary with the following fields: pid_title, pid_title_mask, pid_body, pid_body_mask, candidate_titles, candidate_titles_mask, candidate_body, candidate_body_mask, labels. The candidate titles in turn consist of the index tensors for each candidate question (one positive question followed by random questions) concatenated and similarly for the candidate bodies. Finally the labels indicate which questions are positive.
'''	
def get_dev_data(path, id_to_tensors):
	dev_data = []
	data = read_annotations(path, K_neg=-1, prune_pos_cnt=-1)
	for (pid, qid, qlabels) in data:
		if pid not in id_to_tensors:
			continue
		(pid_title, pid_title_mask, pid_body, pid_body_mask) = id_to_tensors[pid]
		candidate_titles = torch.cat([torch.unsqueeze(id_to_tensors[x][0],0) for x in qid])
		candidate_titles_mask = torch.cat([torch.unsqueeze(id_to_tensors[x][1],0) for x in qid])
		candidate_body = torch.cat([torch.unsqueeze(id_to_tensors[x][2],0) for x in qid])
		candidate_body_mask = torch.cat([torch.unsqueeze(id_to_tensors[x][3],0) for x in qid])
		if len(dev_data) is 0 or candidate_titles.size() == dev_data[0]['candidate_titles'].size():
			dev_data.append({
								'pid_title': pid_title,
								'pid_title_mask': pid_title_mask,
								'pid_body': pid_body,
								'pid_body_mask': pid_body_mask,
								'candidate_titles': candidate_titles,
								'candidate_titles_mask': candidate_titles_mask,
								'candidate_body': candidate_body,
								"candidate_body_mask": candidate_body_mask,
								'labels': torch.LongTensor(qlabels)
								})
	
		
	return dev_data
	
def featurize(path):
	id_to_tfidf = {}
	id_to_idx = {}
	corpus = []
	
	idx = 0
	with open(path) as f:
		for line in f.readlines():
			id = line.split('\t')[0]
			words = ' '.join(line.split('\t')[1:])
			corpus.append(words)
			id_to_idx[id] = idx
			idx = idx + 1
	
	vectorizer = TfidfVectorizer()
	tfidf = vectorizer.fit_transform(corpus).toarray()
	for id in id_to_idx:
		id_to_tfidf[id] = torch.FloatTensor(tfidf[id_to_idx[id]])
	return id_to_tfidf
	
def read_annotations_android(sim_path, rand_path):
	ids = []
	id_to_cand = {}
	id_to_labels = {}
	
	with open(sim_path) as f:
		for line in f.readlines():
			(id, cand) = line.split()
			if id not in ids:
				ids.append(id)
				id_to_cand[id] = [cand]
				id_to_labels[id] = [1]
			else:
				if cand not in id_to_cand[id]:
					id_to_cand[id].append(cand)
					id_to_labels[id].append(1)
	
	with open(rand_path) as f:
		for line in f.readlines():
			(id, cand) = line.split()
			if id not in ids:
				continue
			else:
				if cand not in id_to_cand[id]:
					id_to_cand[id].append(cand)
					id_to_labels[id].append(0)
	
	lst = []
	for id in ids:
		lst.append((id, id_to_cand[id], id_to_labels[id]))
	return lst
