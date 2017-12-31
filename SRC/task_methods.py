"""
Module responsible for task methods (i.e., the neural networks that will be explained).
"""

import numpy as np
np.random.seed(123)
import keras
import _pickle

from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.metrics import *
from corpora import *
from util import *

from explanation_methods import *

class TaskMethod:
	BATCH_SIZE = 8

	def __init__(self, architecture, corpus):
		self.architecture = architecture
		self.corpus = corpus
		
		if not self.architecture in ("LSTM", "GRU", "CNN"):
			raise Exception("Unknown architecture", self.architecture)
	
	def predict_dataset(self, dataset):
		"""
		Return a list of class predictions for dataset
		"""

		generator = self.corpus.get_generator(dataset, self.BATCH_SIZE, shuffle = False)
		spe = self.corpus.get_steps_per_epoch(dataset, self.BATCH_SIZE)

		return self.predict_from_generator(generator, spe)

	def predict_from_generator(self, generator, spe):
		bar = ProgressBar()

		predictions = []
		
		for _ in bar(list(range(spe))):
			x, y = next(generator)
			tmp = self.model.predict_on_batch(x).argmax(-1) # (batch_size,)
			predictions.extend(tmp) # (n_samples,)

		return predictions

	def build(self):
		"""
		Build and compile the keras model
		"""
		self.input = Input((None,))
		self.corpus.load_select_if_necessary(("embeddings", "worddict", "classdict"))

		self.embedding = Embedding(\
			input_dim = self.corpus.FREQCAP, 
			output_dim = self.corpus.EMB_SIZE, 
			mask_zero = self.architecture != "CNN", # Conv1D does not support masking
			name = "embedding", 
			weights = self.corpus.embeddings)
		
		self.dropout1 = Dropout(self.corpus.DROPOUT)

		if self.architecture == "CNN":
			self.main = Sequential(name = "main")
			self.main.add(\
				Conv1D(\
					input_shape = (None, self.corpus.EMB_SIZE), 
					filters = self.corpus.HIDDEN_SIZE, 
					kernel_size = 5, 
					activation = "relu", 
					padding = "same"))
			self.main.add(GlobalMaxPooling1D())

		else:
			mainclass = {"LSTM": LSTM, "GRU": GRU}[self.architecture] # like eval()

			self.main = Bidirectional(\
				mainclass(\
					units = self.corpus.HIDDEN_SIZE // 2, 
					recurrent_dropout = self.corpus.DROPOUT), 
				merge_mode = "concat", name = "main")
		
		self.dropout2 = Dropout(self.corpus.DROPOUT)
		self.dense = Dense(units = len(self.corpus.classdict), activation = "linear")
		self.softmax = Activation("softmax")

		self.output = self.softmax(self.dense(self.dropout2(self.main(self.dropout1(self.embedding(self.input))))))

		self.model = Model([self.input], [self.output])
		self.model.compile(\
			optimizer = "adam", 
			loss = "sparse_categorical_crossentropy", 
			metrics = ["accuracy"])

	def train(self, modelpath, csvpath):
		"""
		Train the task method
		
		modelpath: path where model should be stored
		csvpath: path where logging information can be stored
		"""
		generators = {dset: self.corpus.get_generator(dset, 8, shuffle = dset == "train") \
			for dset in self.corpus.DATASETS}
		spe = {dset: self.corpus.get_steps_per_epoch(dset, 8) \
			for dset in self.corpus.DATASETS}
		self.model.summary()

		# Early Stopping: stops training after a certain number of epochs without improvement of dev set accuracy
		# CSV logger: write some stats to CSV file
		# Model Checkpoint: Store the model if it is the best so far
		# Reduce LR On Pleateau: halve learning rate if dev set accuracy does not improve
		callbacks = [\
			EarlyStopping(patience = self.corpus.PATIENCE, monitor = "val_acc", verbose = 1),
			CSVLogger(csvpath),
			ModelCheckpoint(modelpath, save_best_only = True, monitor = "val_acc", verbose = 1),
			ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, min_lr=0, verbose = 1)]

		# train for 1000 epochs (read: unlimited epochs) until Early Stopping kicks in
		self.model.fit_generator(\
			generators["train"], 
			epochs = 1000, 
			steps_per_epoch = spe["train"],
			validation_data = generators["dev"], 
			validation_steps = spe["dev"], 
			callbacks = callbacks, 
			verbose = 0)
	
	def load(self, modelpath):
		self.model = load_model(modelpath)
		
	def make_explanation_method(self, score):
		"""
		Returns the desired explanation method, which will be applied to 'self'
		"""

		if score == "beta":
			return ScoreModelBeta(self)
		if score == "gamma":
			return ScoreModelGamma(self)
		if score == "omission":
			return ScoreModelErasure(self)
		if score == "grad_raw_l2":
			return ScoreModelGradientRaw(self, "l2")
		if score == "grad_raw_dot":	
			return ScoreModelGradientRaw(self, "dot")
		if score == "grad_prob_l2":
			return ScoreModelGradientProb(self, "l2")
		if score == "grad_prob_dot":
			return ScoreModelGradientProb(self, "dot")
		if score == "lime_class":
			return ScoreLimeClass(self)
		if score == "lime_prob":
			return ScoreLimeProb(self)
		if score == "lime_raw":
			return ScoreLimeRaw(self)
		if score == "lrp":
			return ScoreLRP(self)

		raise Exception("Unknown score", score)
		

	def evaluate(self):
		print("Evaluating on dev and test data")
		print("Architecture:", self.architecture)

		generators = {dset: self.corpus.get_generator(dset, self.BATCH_SIZE, shuffle = False) \
			for dset in ("dev", "test")}
		spe = {dset: self.corpus.get_steps_per_epoch(dset, self.BATCH_SIZE) \
			for dset in generators}
		results = {dset: self.model.evaluate_generator(generators[dset], spe[dset]) \
			for dset in generators}

		for dset in generators:
			print(dset)
			for name, value in zip(self.model.metrics_names, results[dset]):
				print(name, value)
