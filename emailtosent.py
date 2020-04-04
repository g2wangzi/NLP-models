# load data
from nltk import sent_tokenize
from os import listdir
from bert_serving.client import BertClient
import numpy as np

def email2sentence(filename):

	sentences = []
	file = open(filename, 'rt', encoding = "ISO-8859-1")
	# sender = next(file)
	# next(file)
	# next(file)# skipping send time and receiver Bush
	# subject = next(file)
	# sentences.append(sender)
	# sentences.append(subject)
	text = file.read()
	file.close()
	# split into sentences

	sentences += sent_tokenize(text)

	return [sentence for sentence in sentences if len(sentence) > 1]

i = 0
sentences = []
sentence_file_dict = {}

for dirname in listdir("./athome4small/"):
	for filename in listdir("./athome4small/" + dirname):
		sentences_currentfile = email2sentence("./athome4small/" + dirname + "/" + filename)
		sentences += sentences_currentfile
		for sentence in sentences_currentfile:
			if sentence in sentence_file_dict:
				sentence_file_dict[sentence].append(filename)
			else:
				sentence_file_dict[sentence] = [filename]
		i += 1

bc = BertClient()
sentence_vec = bc.encode(sentences)

query = "concerning the space industry, the space program, space travel (whether manned or unmanned, public or private), and the study or exploration of space in Florida"
query_vec = bc.encode([query])[0]
score = np.sum(query_vec * sentence_vec, axis=1) / np.linalg.norm(sentence_vec, axis=1)
topk_idx = np.argsort(score)[::-1][:200]

for idx in topk_idx:
	print(sentences[idx])
	print(sentence_file_dict[sentences[idx]])


