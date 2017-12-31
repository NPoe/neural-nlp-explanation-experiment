"""
Module responsible for preparing, storing and handling data.
"""

import numpy as np
import os
import _pickle
import json

from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.datasets import fetch_20newsgroups

from util import *

def get_corpus_object(corpus_name):
	"""
	Return the correct corpus object based on its name
	"""
	if corpus_name == "yelp":
		return CorpusYelp(make_storagedir(corpus_name), GLOVEPATH, *JSONS)
		
	elif corpus_name == "newsgroup":
		return CorpusNewsgroup(make_storagedir(corpus_name), GLOVEPATH)

	raise Exception("Unknown corpus", corpus_name)

class Corpus:
	"""
	Corpus parent class
	"""

	DATASETS = ("test", "train", "dev", "hybrid")
	FREQCAP = 50000 # words with a frequency rank above this number are mapped to __oov__
	MAXLENGTH = 1000 # number of words above which documents are trimmed
	FILENAMES = ("embeddings", "classdict", "worddict", "X", "Y", "GT", "tokenized_documents", "raw_documents")
	EMB_SIZE = 300 # embedding size (same for both corpora)
	HIDDEN_SIZE = 150 # hidden size (same for both corpora)
	DROPOUT = 0.5 # dropout (same for both corpora)
	HYBRID_LENGTH = 10 # length of hybrid documents, in sentences
	
	def __init__(self, storagedir, embeddingpath):
		self.storagedir = storagedir
		self.embeddingpath = embeddingpath
		self.pred = {}

	def prepare(self):
		"""
		Prepare the corpus by storing all necessary files in the storage directory.
		"""
		if len(os.listdir(self.storagedir)):
			raise Exception("There are already files in", self.storagedir + ".", "Delete manually!")
		
		self.worddict = {"__pad__": 0, "__oov__": 1}
		self.classdict = {}
		self.raw_documents, self.tokenized_documents = {}, {}
		self.X, self.Y = {}, {}

		for dataset in self.DATASETS_TMP:
			self.get_raw_data(dataset)
			self.delete_empty_documents(dataset)
			self.tokenize_documents(dataset)

		self.make_classdict()
		self.make_worddict()
		self.make_embeddings()
		self.reverse_dicts()

		for dataset in self.DATASETS_TMP:
			self.make_X(dataset)
			self.shuffle_dataset(dataset)

		if not "dev" in self.X:
			self.split_dev()		
		self.make_hybrid()
		self.store()

	def make_embeddings(self):
		"""
		Preset embedding weights with GloVe pre-trained embeddings (where possible).
		"""
		print("Presetting embedding weights")
			
		np.random.seed(0)
		weights = np.random.uniform(low = -0.05, high = 0.05, size = (self.FREQCAP, self.EMB_SIZE))
		
		counter = 0

		words = []
		weights_tmp = []

		with open(self.embeddingpath) as handle:
			for i, line in enumerate(handle):
				tmp = line.strip()
				if len(tmp) > 0:
					split = tmp.split(" ")
					if split[0] in self.worddict and len(split[1:]) == 300:
						words.append(split[0])
						weights_tmp.append([float(a) for a in split[1:]])
		
		weights_tmp = np.array(weights_tmp)

		for word, column in zip(words, weights_tmp):
			if self.worddict[word] < self.FREQCAP:
				counter += 1
				weights[self.worddict[word],:] = column
		
		print("Set", counter, "of", weights.shape[0], "columns")
		
		if self.EMB_SIZE < weights.shape[-1]:
			print("Reducing dimensionality to", self.EMB_SIZE)
			pca = PCA(self.EMB_SIZE)
			weights = pca.fit_transform(weights)
		
		self.embeddings = [weights]	

	
	def reverse_dicts(self):
		"""
		Reverse class and word dicts; important for printing + sanity checks
		"""
		self.rev_worddict = {self.worddict[word]: word for word in self.worddict}
		self.rev_classdict = {self.classdict[cl]: cl for cl in self.classdict}

	def store(self):
		"""
		Store corpus to its storage directory
		"""
		print("Storing to", self.storagedir)

		for filename in self.FILENAMES:
			with open(os.path.join(self.storagedir, filename), "wb") as handle:
				_pickle.dump(getattr(self, filename), handle)

	def load(self, which):
		"""
		Load a corpus component from its storage directory
		"""
		path = os.path.join(self.storagedir, which)
		print("Loading from", path)
		with open(path, "rb") as handle:
			setattr(self, which, _pickle.load(handle))

	def load_full(self):
		"""
		Load the entire corpus from its storage directory
		"""
		for filename in self.FILENAMES:
			self.load(filename)
		self.reverse_dicts()

	def load_select(self, selected):
		"""
		Load selected components (from list) from corpus storage directory
		"""
		for filename in selected:
			self.load(filename)

		if "worddict" in selected and "classdict" in selected:
			self.reverse_dicts()	

	def get_steps_per_epoch(self, dataset, batchsize):
		"""
		Returns the number of steps that are necessary to generate all samples exactly once.

		dataset: one of 'train', 'dev', 'test', 'hybrid'
		batchsize: batch size that the generator will be working on
		"""
		self.load_if_necessary("X")

		num_samples = len(self.X[dataset])
		if num_samples % batchsize == 0:
			return num_samples // batchsize

		return num_samples // batchsize + 1 # account for the smaller last batch if necessary
		
	def trim_and_pad_batch(self, batch):
		"""
		Trim all samples in a batch to MAXLENGTH and pad them to identical lengths.
		"""
		maxlength = min(self.MAXLENGTH, max([len(x) for x in batch]))
				
		batch = [x[:maxlength] for x in batch]
		batch = [np.concatenate([x, np.zeros(maxlength - x.shape[0])]) for x in batch]

		return batch
	
	def load_if_necessary(self, which):
		"""
		Load corpus component only if it has not yet been loaded
		"""
		if not hasattr(self, which):
			self.load(which)
			
	def load_select_if_necessary(self, selected):
		"""
		Load selected corpus components only if they have not yet been loaded
		"""
		for which in selected:
			self.load_if_necessary(which)

		if "worddict" in selected and "classdict" in selected:
			self.reverse_dicts()


	def get_generator(self, dataset, batchsize, shuffle = False):
		"""
		Returns a generator that will generate (X,Y) pairs for the given dataset.

		dataset: one of 'train', 'dev', 'test', 'hybrid'
		batchsize: batch size that the generator will be working on
		shuffle: if true, the dataset is shuffled at the beginning of every epoch
		"""
		self.load_select_if_necessary(("X", "Y"))
		random_state = np.random.RandomState(0)

		while True:
			indices = list(range(len(self.X[dataset])))
			if shuffle:
				random_state.shuffle(indices)

			X = [self.X[dataset][idx] for idx in indices]
			Y = [self.Y[dataset][idx] for idx in indices]

			for idx in range(0, len(X), batchsize):
				batch_X = X[idx:min(idx + batchsize, len(X))]
				batch_Y = Y[idx:min(idx + batchsize, len(X))]
				batch_X = np.array(self.trim_and_pad_batch(batch_X))

				yield(batch_X, np.array(batch_Y))

	def sanity_check(self):
		"""
		A number of checks to make sure that data is generated correctly
		"""
		self.load_full()
		generators_not_shuffling = {dataset: self.get_generator(dataset, 16, False) for dataset in self.DATASETS}
		generators_shuffling = {dataset: self.get_generator(dataset, 16, True) for dataset in self.DATASETS}
		steps_per_epoch = {dataset: self.get_steps_per_epoch(dataset, 16) for dataset in self.DATASETS}

		# make sure that non-shuffling generators return data in the same order every epoch
		# and that shuffling generators don't
		for dataset in self.DATASETS:
			print(dataset)

			assert len(self.X[dataset]) == len(self.Y[dataset])
			
			for _ in range(50):
				x1, y1 = next(generators_not_shuffling[dataset])
			for _ in range(steps_per_epoch[dataset]):
				x2, y2 = next(generators_not_shuffling[dataset])
			
			assert np.allclose(x1, x2)
			assert np.allclose(y1, y2)

			for _ in range(50):
				x1, y1 = next(generators_shuffling[dataset])
			for _ in range(steps_per_epoch[dataset]):
				x2, y2 = next(generators_shuffling[dataset])
			
			assert x1.shape != x2.shape or not np.allclose(x1, x2)
			
			if dataset != "hybrid":
				assert not np.allclose(y1, y2)

			# display some data
			for k in (6, 77, 99):
				for _ in range(k):
					x, y = next(generators_shuffling[dataset])
				words = [self.rev_worddict[word] for word in x[0] if word > 0]
				label = self.rev_classdict[y[0]]
				text = " ".join(words)
				print(label)
				print(text)
				print()

		print("Hybrid documents")

		generator_hybrid = self.get_generator("hybrid", 1)
		counter = -1
		for k in (55, 66, 999):
			for _ in range(k):
				x, y = next(generator_hybrid)
				counter += 1
			words = [self.rev_worddict[word] for word in x[0] if word > 0]
			labels = ["(" + self.rev_classdict[label] + ")" for label in self.GT[counter]]
			text = " ".join(word + " " + label for word, label in zip(words, labels))
			print(text)
			print()
		
			
		
				
	def delete_empty_documents(self, dataset):
		"""
		Delete any documents that do not contain any words (i.e., that were blank-only).
		
		dataset: one of 'train', 'dev', 'test', 'hybrid'
		"""
		print("Deleting empty documents in", dataset)
		number_documents = len(self.raw_documents[dataset])
		indices = list(filter(lambda x:len(self.raw_documents[dataset][x].strip()), range(number_documents)))

		self.raw_documents[dataset] = [self.raw_documents[dataset][idx] for idx in indices]
		self.Y[dataset] = [self.Y[dataset][idx] for idx in indices]
	
	def tokenize_documents(self, dataset):
		print("Word-tokenizing documents in", dataset)
		self.tokenized_documents[dataset] = [word_tokenize(document) for document in self.raw_documents[dataset]]

	def shuffle_dataset(self, dataset):
		print("Shuffling dataset", dataset)

		indices = list(range(len(self.X[dataset])))
		np.random.seed(0)
		np.random.shuffle(indices)

		self.X[dataset] = [self.X[dataset][idx] for idx in indices]
		self.Y[dataset] = [self.Y[dataset][idx] for idx in indices]
		self.tokenized_documents[dataset] = [self.tokenized_documents[dataset][idx] for idx in indices]
		self.raw_documents[dataset] = [self.raw_documents[dataset][idx] for idx in indices]


	def make_X(self, dataset):
		"""
		Create word index arrays from the tokenized documents.

		The word index arrays serve as input to training/evaluation/relevance scoring.
		"""
		print("Making X", dataset)
		self.X[dataset] = []
		for document in self.tokenized_documents[dataset]:
			array = np.array([self.worddict.get(word, self.worddict["__oov__"]) for word in document])
			self.X[dataset].append(array)
	
	def make_hybrid(self):
		"""
		Create hybrid documents by:
	
		1) sentence-tokenizing the raw documents in the test set
		2) shuffling all sentences
		3) re-concatenating the sentences
		"""
		print("Making hybrid documents")
		self.X["hybrid"] = []
		self.tokenized_documents["hybrid"] = []
		self.GT = []

		all_sentences = []
		for document, label in zip(self.raw_documents["test"], self.Y["test"]):
			sentences = sent_tokenize(document)
			for sentence in sentences:
				all_sentences.append((sentence, label))

		np.random.seed(0)
		np.random.shuffle(all_sentences)

		for i in range(0, len(all_sentences), self.HYBRID_LENGTH):
			batch = all_sentences[i:min(i+self.HYBRID_LENGTH, len(all_sentences))]

			hybrid_tokenized_document = []
			hybrid_X = []
			hybrid_labels = []

			for sentence, label in batch:
				for word in word_tokenize(sentence):
					hybrid_tokenized_document.append(word)
					hybrid_X.append(self.worddict.get(word, self.worddict["__oov__"]))
					hybrid_labels.append(label)

			self.X["hybrid"].append(np.array(hybrid_X))
			self.tokenized_documents["hybrid"].append(hybrid_tokenized_document)
			self.GT.append(np.array(hybrid_labels))

		self.Y["hybrid"] = np.zeros(len(self.X["hybrid"])) # pseudo-labels, we won't do anything with these

		print("Created", len(self.X["hybrid"]), "hybrid documents from", len(self.X["test"]), "test documents")

	
	def make_word_to_freq(self):
		"""
		Map all words in the corpus to their absolute frequency
		"""
		word_to_freq = {}
		documents = self.tokenized_documents["train"]
		for document in documents:
			for word in document:
				if not word in self.worddict: # make sure we have not found one of the pre-defined words
					word_to_freq[word] = word_to_freq.get(word, 0) + 1
		
		return word_to_freq
			
	def make_worddict(self):
		"""
		Create a dictionary that maps word types to their frequency rank (e.g., 'and' -> 6)
		"""
		print("Making word dictionary")
		word_to_freq = self.make_word_to_freq()
		words = list(word_to_freq.keys())
		words.sort() # sort alphabetically first to avoid non-deterministic ordering of words with the same frequency
		words.sort(key = lambda x:word_to_freq[x], reverse = True)

		for word in words[:self.FREQCAP-len(self.worddict)]:
			self.worddict[word] = len(self.worddict)
		
		print("Word dictionary size:", len(self.worddict))
	
			
class CorpusNewsgroup(Corpus):
	DATASETS_TMP = ("test", "train") # names of the datasets that are initially downloaded
	PATIENCE = 25 # number of epochs to wait for early stopping
	NAME = "newsgroup"

	def __init__(self, storagedir, embeddingpath = None, *args):
		super(CorpusNewsgroup, self).__init__(storagedir, embeddingpath)
		self.fetched = {}

	def make_classdict(self):
		"""
		Make a dictionary that maps class names to class indices (e.g., 'sci.med' -> 16)
		"""
		target_names = self.fetched["train"].target_names
		self.classdict = {target_names[idx]: idx for idx in range(len(target_names))}

	def get_raw_data(self, dataset):
		"""
		Download raw data for one of 'train', 'test'
		"""
		print("Getting raw data for", dataset)
		self.fetched[dataset] = fetch_20newsgroups(remove = ('headers', 'footers', 'quotes'), subset = dataset)
		self.raw_documents[dataset] = self.fetched[dataset].data
		self.Y[dataset] = self.fetched[dataset].target
	
	def split_dev(self):
		"""
		Randomly split test set into a development set and test set.
		"""
		print("Splitting test set into dev and test set")

		old_length = len(self.X["test"])
		indices = list(range(old_length))

		np.random.seed(0)
		np.random.shuffle(indices)
		
		split = int(len(indices) * 0.5)

		split_indices = {"test": indices[:split], "dev": indices[split:]}
	
		for dataset in ("dev", "test"):
			self.X[dataset] = [self.X["test"][idx] for idx in split_indices[dataset]]
			self.Y[dataset] = [self.Y["test"][idx] for idx in split_indices[dataset]]
			self.raw_documents[dataset] = [self.raw_documents["test"][idx] for idx in split_indices[dataset]]
			self.tokenized_documents[dataset] = [self.tokenized_documents["test"][idx] for idx in split_indices[dataset]]
		
		print("Split test set with", old_length, "samples into", len(self.X["test"]), "/", len(self.X["dev"]), "samples")


class CorpusYelp(Corpus):
	DATASETS_TMP = ("test", "dev", "train")
	PATIENCE = 5 # since the yelp corpus takes longer to train, we only wait for five epochs before early stopping
	NAME = "yelp"

	def __init__(self, storagedir, embeddingpath = None, trainjson = None, devjson = None, testjson = None):
		super(CorpusYelp, self).__init__(storagedir, embeddingpath)
		self.jsons = {"train": trainjson, "test": testjson, "dev": devjson}
		
	def make_classdict(self):
		self.classdict = {"negative": 0, "positive": 1}

	def get_raw_data(self, dataset):
		"""
		Read raw data from the json file associated with <dataset>

		(path to be set in config.py)
		"""
		print("Getting raw data for", dataset)
		self.raw_documents[dataset] = []
		self.Y[dataset] = []

		with open(self.jsons[dataset]) as handle:
			for line in handle:
				json_obj = json.loads(line)
				stars = json_obj["stars"]
				if stars != 3:
					self.raw_documents[dataset].append(json_obj["text"])
					self.Y[dataset].append(int(stars > 3))
