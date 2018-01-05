import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import data_processor as dp
import time
from torch.nn.functional import tanh
from torch.nn.functional import relu
from torch.autograd import Variable
from evaluate import Evaluation
from meter import AUCMeter

class CNN(nn.Module):
	def __init__(self, output_size, kernel_width, embeddings, dropout):
		super(CNN, self).__init__()
		self.kernel_width = kernel_width
		self.output_size = output_size

		vocab_size, embed_dim = embeddings.shape
		self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
		self.embedding_layer.weight.data = torch.from_numpy( embeddings )

		self.conv = nn.Conv1d(embed_dim, output_size, kernel_width, stride=1)
		self.dropout = nn.Dropout(p = dropout)
		
	def forward(self, input):
		(batch_size, samples, length) = input.size()
		x = input.view(batch_size * samples, length)
		x = self.embedding_layer(x)
		x = x.float()
		x = self.dropout(x)
		x = torch.transpose(x, 1, 2) 
		x = tanh(self.dropout(self.conv(x)))
		x = x.view(batch_size, samples, self.output_size, length - self.kernel_width + 1)
		l2 = torch.norm(x, 2, dim=2, keepdim=True).expand_as(x)
		x = x / l2.clamp(min = 1e-8)
	
		return x
		
class LSTM(nn.Module):
	def __init__(self, output_size, embeddings, dropout):
		super(LSTM, self).__init__()
		self.output_size = output_size

		vocab_size, embed_dim = embeddings.shape
		self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
		self.embedding_layer.weight.data = torch.from_numpy( embeddings )
		self.embedding_layer.weight.requires_grad = False
		
		self.lstm = nn.LSTM(embed_dim, output_size // 2, num_layers = 1,
							bias = True, batch_first = True, dropout = dropout,
							bidirectional = True)
		self.dropout = nn.Dropout(p = dropout)
				
	def forward(self, input):
		(batch_size, samples, length) = input.size()
		x = input.view(batch_size * samples, length)
		x = self.embedding_layer(x)
		x = x.float()
		x = self.dropout(x)
		output, hn = self.lstm(x) 
		x = torch.transpose(output, 1, 2)
		x = x.contiguous().view(batch_size, samples, self.output_size, length)
		l2 = torch.norm(x, 2, dim=2, keepdim=True).expand_as(x)
		x = x / l2.clamp(min = 1e-8)
		return x
		
class DomainClassifier(nn.Module):
	def __init__(self, embeddings, hidden1, hidden2):
		super(DomainClassifier, self).__init__()
		self.linear1 = nn.Linear(embeddings, hidden1)
		self.linear2 = nn.Linear(hidden1, hidden2)
		self.out = nn.Linear(hidden2, 2)
		
	def forward(self, input):
		hidden = relu(self.linear1(input))
		hidden2 = relu(self.linear2(hidden))
		out = self.out(hidden2)
		return out
		
def cosine_sim(tensor1, tensor2, dim):
	prod = torch.sum(tensor1 * tensor2, dim)
	norm1 = torch.norm(tensor1, 2, dim)
	norm2 = torch.norm(tensor2, 2, dim)
	return prod/(norm1 * norm2).clamp(min=.00001)
		
def loss(good_tensor, cand_tensor, delta):
	sim = cosine_sim(good_tensor.expand_as(cand_tensor), cand_tensor, dim=2)
	sim_sim = sim[:, 0]
	rand_sim = sim[:, 1:]
	max_rand_sim = torch.max(rand_sim, dim=1)[0]
	loss = relu(max_rand_sim - sim_sim + delta)
	return torch.mean(loss)
		
def train_model(train_data, dev_data, model, transfer=False):
	model.cuda()
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
	model.train()

	prev_loss = 1000
	prev_time = time.time()
	best_mrr = 0
	for epoch in range(50):
		print("****************************************")
		print("Epoch", epoch + 1)

		loss = run_epoch(train_data, True, model, optimizer, transfer)
		print("Trained in:", time.time() - prev_time)
		print("Loss:", loss)
		
		if transfer:
			AUC = run_epoch(dev_data, False, model, optimizer, transfer)
			print("AUC:", AUC, "(Benchmark: 0.6)")
		else:
			(MAP, MRR, P1, P5) = run_epoch(dev_data, False, model, optimizer, transfer)
			print("MAP:", MAP, "(Benchmark: 52.0 dev, 56.0 test)")
			print("MRR:", MRR, "(Benchmark: 66.0 dev, 68.0 test)")
			print("P1:", P1, "(Benchmark: 51.9 dev, 53.8 test)")
			print("P5:", P5, "(Benchmark: 42.1 dev, 42.5 test)")
		torch.save(model, "model{}".format(epoch))
		
		print("Evaluated in:", time.time() - prev_time)
		prev_time = time.time()
		
		# Early stopping
		if epoch > 10 and prev_loss - loss < 0.001:
			break
			
		prev_loss = loss
		
def run_epoch(data, is_training, model, optimizer, transfer=False):
	
	# Make batches
	data_loader = torch.utils.data.DataLoader(
		data,
		batch_size=10,
		shuffle=True,
		num_workers=4,
		drop_last=False)

	losses = []
	actual = []
	expected = []

	if is_training:
		model.train()
	else:
		model.eval()
	
	for batch in data_loader:
		# Unpack training instances
		pid_title = torch.unsqueeze(Variable(batch['pid_title']), 1).cuda() # Size: batch_size x 1 x title_length=40
		pid_title_mask = torch.unsqueeze(Variable(batch['pid_title_mask']), 1).cuda() # Size: batch_size x 1 x title_length=40
		pid_body = torch.unsqueeze(Variable(batch['pid_body']), 1).cuda() # Size: batch_size x 1 x body_length=100
		pid_body_mask = torch.unsqueeze(Variable(batch['pid_body_mask']), 1).cuda() # Size: batch_size x 1 x body_length=100
		candidate_title = Variable(batch['candidate_titles']).cuda() # Size: batch_size x # candidates (21 in training) x title_length=40
		candidate_title_mask = Variable(batch['candidate_titles_mask']).cuda() # Size: batch_size x # candidates (21 in training) x title_length=40
		candidate_body = Variable(batch['candidate_body']).cuda() # Size: batch_size x # candidates (21 in training) x body_length=100
		candidate_body_mask = Variable(batch['candidate_body_mask']).cuda() # Size: batch_size x # candidates (21 in training) x body_length=40
		
		if is_training:
			optimizer.zero_grad()
		
		# Run text through model
		pid_title = model(pid_title) # batch_size x 1 x output_size=500 x title_length=40(-kernel_size+1 if CNN)
		pid_body = model(pid_body) # batch_size x 1 x output_size=500 x body_length=100(-kernel_size+1 if CNN)
		candidate_title = model(candidate_title) # batch_size x # candidates (21 in training) x output_size=500 x title_length=40(-kernel_size+1 if CNN)
		candidate_body = model(candidate_body) # batch_size x # candidates (21 in training) x output_size=500 x body_length=100(-kernel_size+1 if CNN)
		
		pid_title_mask = torch.unsqueeze(pid_title_mask, 2).expand_as(pid_title) # batch_size x 1 x output_size=500 x title_length=40(-kernel_size+1 if CNN)
		pid_body_mask = torch.unsqueeze(pid_body_mask, 2).expand_as(pid_body) # batch_size x 1 x output_size=500 x body_length=100(-kernel_size+1 if CNN)
		candidate_title_mask = torch.unsqueeze(candidate_title_mask, 2).expand_as(candidate_title)# batch_size x # candidates (21 in training) x output_size=500 x title_length=40(-kernel_size+1 if CNN)
		candidate_body_mask = torch.unsqueeze(candidate_body_mask, 2).expand_as(candidate_body) # batch_size x # candidates (21 in training) x output_size=500 x body_length=100(-kernel_size+1 if CNN)

		good_title = torch.sum(pid_title * pid_title_mask, 3) # batch_size x 1 x output_size=500
		good_body = torch.sum(pid_body * pid_body_mask, 3) # batch_size x 1 x output_size=500
		cand_titles = torch.sum(candidate_title * candidate_title_mask, 3) # batch_size x # candidates (21 in training) x output_size=500
		cand_bodies = torch.sum(candidate_body * candidate_body_mask, 3) # batch_size x # candidates (21 in training) x output_size=500
		
		good_tensor = (good_title + good_body)/2 # batch_size x 1 x output_size=500
		cand_tensor = (cand_titles + cand_bodies)/2 # batch_size x # candidates (21 in training) x output_size=500
		
		if is_training:
			l = loss(good_tensor, cand_tensor, 1.0)
			l.backward()
			losses.append(l.cpu().data[0])
			optimizer.step()
		else:
			similarity = cosine_sim(good_tensor.expand_as(cand_tensor), cand_tensor, dim=2)
			if transfer:
				similarity = torch.FloatTensor(similarity.data.cpu().numpy())
			else:
				similarity = similarity.data.cpu().numpy()
			if transfer:
				labels = batch['labels']
			else:
				labels = batch['labels'].numpy()
			def predict(sim, labels):
				predictions = []
				for i in range(sim.shape[0]):
					sorted_cand = (-sim[i]).argsort()
					predictions.append(labels[i][sorted_cand])
				return predictions
			if transfer:
				for sim in similarity:
					actual.append(sim)
				expected.extend(labels.view(-1))
			else:
				l = predict(similarity, labels)
				losses.extend(l)

	if is_training:
		avg_loss = np.mean(losses)
		return avg_loss
	else:
		if transfer:
			auc = AUCMeter()
			auc.reset()
			auc.add(torch.cat(actual), torch.LongTensor(expected))
			return auc.value(max_fpr=0.05)
		else:
			e = Evaluation(losses)
			MAP = e.MAP()*100
			MRR = e.MRR()*100
			P1 = e.Precision(1)*100
			P5 = e.Precision(5)*100
			return (MAP, MRR, P1, P5)
			
def train_adversarial_model(train, dev, encoder, classifier):
	encoder.cuda()
	classifier.cuda()
	encoder_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=0.0001)
	classifier_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=-0.0001)

	prev_loss = 1000
	prev_time = time.time()
	best_mrr = 0
	for epoch in range(50):
		print("****************************************")
		print("Epoch", epoch + 1)

		(loss, bce_loss) = run_adversarial_epoch(train, True, encoder, encoder_optimizer, classifier, classifier_optimizer)
		print("Trained in:", time.time() - prev_time)
		print("Loss:", loss)
		print("BCE loss:", bce_loss)
		
		AUC = run_adversarial_epoch(dev, False, encoder, encoder_optimizer, classifier, classifier_optimizer)
		print("AUC:", AUC, "(Benchmark: 0.66)")
		torch.save(encoder, "encoder{}".format(epoch))
		torch.save(classifier, "classifier{}".format(epoch))
		
		print("Evaluated in:", time.time() - prev_time)
		prev_time = time.time()
		
		# Early stopping
		if epoch > 10 and prev_loss - loss < 0.001:
			break
			
		prev_loss = loss
		
def run_adversarial_epoch(data, is_training, encoder, encoder_optimizer, classifier, classifier_optimizer):
	
	# Make batches
	data_loader = torch.utils.data.DataLoader(
		data,
		batch_size=10,
		shuffle=True,
		num_workers=4,
		drop_last=False)

	losses = []
	bce_losses = []
	actual = []
	expected = []

	if is_training:
		encoder.train()
		classifier.train()
	else:
		encoder.train()
		classifier.train()
	
	for batch in data_loader:
		# Unpack training instances
		pid_title = torch.unsqueeze(Variable(batch['pid_title']), 1).cuda() # Size: batch_size x 1 x title_length=40
		pid_title_mask = torch.unsqueeze(Variable(batch['pid_title_mask']), 1).cuda() # Size: batch_size x 1 x title_length=40
		pid_body = torch.unsqueeze(Variable(batch['pid_body']), 1).cuda() # Size: batch_size x 1 x body_length=100
		pid_body_mask = torch.unsqueeze(Variable(batch['pid_body_mask']), 1).cuda() # Size: batch_size x 1 x body_length=100
		candidate_title = Variable(batch['candidate_titles']).cuda() # Size: batch_size x # candidates (21 in training) x title_length=40
		candidate_title_mask = Variable(batch['candidate_titles_mask']).cuda() # Size: batch_size x # candidates (21 in training) x title_length=40
		candidate_body = Variable(batch['candidate_body']).cuda() # Size: batch_size x # candidates (21 in training) x body_length=100
		candidate_body_mask = Variable(batch['candidate_body_mask']).cuda() # Size: batch_size x # candidates (21 in training) x body_length=40
		if is_training:
			android_title = torch.unsqueeze(Variable(batch['android_title']), 1).cuda()
			android_title_mask = torch.unsqueeze(Variable(batch['android_title_mask']), 1).cuda()
			android_body = torch.unsqueeze(Variable(batch['android_body']), 1).cuda()
			android_body_mask = torch.unsqueeze(Variable(batch['android_body_mask']), 1).cuda()
		sz = pid_title.size()[0]
		
		if is_training:
			encoder_optimizer.zero_grad()
			classifier_optimizer.zero_grad()
		
		# Run text through model
		pid_title = encoder(pid_title) # batch_size x 1 x output_size=500 x title_length=40(-kernel_size+1 if CNN)
		pid_body = encoder(pid_body) # batch_size x 1 x output_size=500 x body_length=100(-kernel_size+1 if CNN)
		candidate_title = encoder(candidate_title) # batch_size x # candidates (21 in training) x output_size=500 x title_length=40(-kernel_size+1 if CNN)
		candidate_body = encoder(candidate_body) # batch_size x # candidates (21 in training) x output_size=500 x body_length=100(-kernel_size+1 if CNN)
		if is_training:
			android_title = encoder(android_title)
			android_body = encoder(android_body)
		
		pid_title_mask = torch.unsqueeze(pid_title_mask, 2).expand_as(pid_title) # batch_size x 1 x output_size=500 x title_length=40(-kernel_size+1 if CNN)
		pid_body_mask = torch.unsqueeze(pid_body_mask, 2).expand_as(pid_body) # batch_size x 1 x output_size=500 x body_length=100(-kernel_size+1 if CNN)
		candidate_title_mask = torch.unsqueeze(candidate_title_mask, 2).expand_as(candidate_title)# batch_size x # candidates (21 in training) x output_size=500 x title_length=40(-kernel_size+1 if CNN)
		candidate_body_mask = torch.unsqueeze(candidate_body_mask, 2).expand_as(candidate_body) # batch_size x # candidates (21 in training) x output_size=500 x body_length=100(-kernel_size+1 if CNN)
		if is_training:
			android_title_mask = torch.unsqueeze(android_title_mask, 2).expand_as(android_title)
			android_body_mask = torch.unsqueeze(android_body_mask, 2).expand_as(android_body)

		good_title = torch.sum(pid_title * pid_title_mask, 3) # batch_size x 1 x output_size=500
		good_body = torch.sum(pid_body * pid_body_mask, 3) # batch_size x 1 x output_size=500
		cand_titles = torch.sum(candidate_title * candidate_title_mask, 3) # batch_size x # candidates (21 in training) x output_size=500
		cand_bodies = torch.sum(candidate_body * candidate_body_mask, 3) # batch_size x # candidates (21 in training) x output_size=500
		
		if is_training:
			android_title = torch.sum(android_title * android_title_mask, 3)
			android_body = torch.sum(android_body * android_body_mask, 3)
		
		good_tensor = (good_title + good_body)/2 # batch_size x 1 x output_size=500
		cand_tensor = (cand_titles + cand_bodies)/2 # batch_size x # candidates (21 in training) x output_size=500
		if is_training:
			android_tensor = (android_title + android_body)/2
		
		if is_training:
			good_domain = classifier(good_tensor.view(sz, -1))
			android_domain = classifier(android_tensor.view(sz, -1))
			
			softmax = nn.Softmax(dim=1)
			good_dist = softmax(good_domain)
			android_dist = softmax(android_domain)
		
			dists = torch.cat((good_dist, android_dist)).clamp(min=0.0001, max=0.9999)
			expected = Variable(torch.FloatTensor([0] * sz + [1] * sz)).cuda()
			
			bce_loss = torch.nn.BCELoss()(dists[:,0], expected)
			l = loss(good_tensor, cand_tensor, 1.0) - 0.01 * bce_loss
			l.backward()
			losses.append(l.cpu().data[0])
			bce_losses.append(bce_loss.cpu().data[0])
			encoder_optimizer.step()
			classifier_optimizer.step()
		else:
			similarity = cosine_sim(good_tensor.expand_as(cand_tensor), cand_tensor, dim=2)
			similarity = torch.FloatTensor(similarity.data.cpu().numpy())
			labels = batch['labels']
			def predict(sim, labels):
				predictions = []
				for i in range(sim.shape[0]):
					sorted_cand = (-sim[i]).argsort()
					predictions.append(labels[i][sorted_cand])
				return predictions
			for sim in similarity:
				actual.append(sim)
			expected.extend(labels.view(-1))

	if is_training:
		avg_loss = np.mean(losses)
		avg_bce_loss = np.mean(bce_losses)
		return avg_loss, avg_bce_loss
	else:
		auc = AUCMeter()
		auc.reset()
		auc.add(torch.cat(actual), torch.LongTensor(expected))
		return auc.value(max_fpr=0.05)
