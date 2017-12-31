"""
Module responsible for explanation methods.
"""

import numpy as np
np.random.seed(123)
import keras
import _pickle
from keras.models import *
from keras.layers import *
from keras.wrappers import *
from corpora import *
from util import *
from progressbar import ProgressBar

def get_pred_generator(dataset, batchsize, architecture, corpus):
	"""
	Generator that will return arrays of size (batch_size,) with classes predicted by <architectur> for
	data set <dataset> of corpus <corpus>.
	"""
	with open(make_predpath(dataset, architecture, corpus.NAME), "rb") as handle:
		pred = _pickle.load(handle)
	while True:
		for idx in range(0, len(pred), batchsize):
			yield np.array(pred[idx:min(idx+batchsize, len(pred))])


class Score:
	"""
	Parent class for all explanation methods.
	"""
	BATCH_SIZE = 1

	def __init__(self, orig):
		self.orig = orig # pointer to the tas kmethod

	def score_dataset(self, dataset, pred_only = False):
		"""
		Produce relevance scores for all samples of the dataset.

		dataset: one of 'test', 'hybrid'
		pred_only: if true, relevance scores are only calculated for the class predicted by the task method
		"""
		generator = self.orig.corpus.get_generator(dataset, self.BATCH_SIZE, shuffle = False)
		spe = self.orig.corpus.get_steps_per_epoch(dataset, self.BATCH_SIZE)
		if pred_only:
			pred_generator = get_pred_generator(dataset, self.BATCH_SIZE, self.orig.architecture, self.orig.corpus)
		else:
			pred_generator = None
		tmp = self.score_from_generator(generator, spe, pred_generator = pred_generator)
		return tmp
	
	def score_from_generator(self, generator, spe, pred_generator):
		"""
		Produce relevance scores from a generator

		generator: generator that returns (X,Y) tuples where X has the shape (batch_size, num_words)
		spe: steps per epoch
		pred_generator: generator returning arrays of target classes of the shape (batch_size,); can be None
		"""
		bar = ProgressBar()

		scores = []

		corr = 0
		total = 0
		for _ in bar(list(range(spe))):
			x, y = next(generator)
			if pred_generator is None:
				pred = None
			else:
				pred = next(pred_generator)
			
			score = self.score(x, pred)
			scores.extend(score)
		return scores
	
	def get_layer(self, layertype):
		"""
		Utility function that returns keras layer of a certain type.
		We can do this because there is only ever one layer of a particular type in our models,
		otherwise, this is a bad idea!

		layertype: layer class (e.g., Embedding, Bidirectional, Dense)
		"""
		tmp = list(filter(lambda x:type(x) == layertype, self.orig.model.layers))
		assert len(tmp) == 1
		return tmp[0]

class ScoreModel(Score):
	def score(self, x, pred):
		"""
		Return relevance scores for x.
		
		pred: if none, return scores for all possible classes
		else, assume that pred is a list of target classes
		"""

		# n.b. with all children of this class, calling score_k does not result in a speed-up,
		# since all target classes are calculated anyway
		tmp = self.score_model.predict(x)
		if not pred is None:
			# if we only look at predicted classes, get the correct indices
			tmp = np.array([t[:,p:p+1] for t,p in zip(tmp, pred)])
		# cut off any zero padding
		tmp = [np.array([tmp[i][j] for j in range(tmp.shape[1]) if x[i][j] != 0]) for i in range(tmp.shape[0])]
		return tmp
			
	
	def check(self):
		"""
		Basic sanity check (e.g., output shapes ...)
		"""
		assert self.score_model.output_shape == (None,) + self.orig.model.output_shape

		this_weights = self.score_model.get_weights()
		orig_weights = self.get_orig_weights()
		
		assert len(this_weights) == len(orig_weights)
		assert all([np.allclose(x,y) for x,y in zip(this_weights, orig_weights)])
		
		_ = self.score_model.predict(np.array([[1,2,3,4,5,0,0], [4,5,6,7,8,9,0]]))

		print("Score model passed all checks")
		
		self.score_model.summary()

class ScoreModelLinear(ScoreModel):
	"""
	Score Model that is linear, meaning that the fully connected layer can be applied after the relevance scoring layer.
	"""
	def build(self):

		embedding = self.get_layer(Embedding)
		
		dense = self.get_layer(Dense)
		dense_config = dense.get_config()
		dense_config["use_bias"] = False # bias is cancelled out in beta, gamma & omission scores

		self.score_model = Sequential()
		self.score_model.add(embedding)
		self.build_inner() # build the relevance scoring layer
		
		self.score_model.add(TimeDistributed(Dense(**dense_config, weights = dense.get_weights()[:1]))) # add dense layer on top

		self.check()

	def get_orig_weights(self):
		return self.orig.model.get_weights()[:-1]

class ScoreModelBetaGamma(ScoreModelLinear):
	"""
	Beta, gamma decomposition scores (Murdoch & Szlam 2017), eq. 11 (gamma)
	"""
	BATCH_SIZE = 8

	def build_inner(self):
		if self.orig.architecture == "CNN":
			raise Exception("Cannot use beta or gamma decomposition on a CNN")
		
		bidir = self.get_layer(Bidirectional)
		rnn = bidir.forward_layer
		rnn_config = rnn.get_config()
		dense = self.get_layer(Dense)
		
		# a bug (?) in keras means that we cannot keep the dropout
		# since dropout is only used in training, this does not make a difference to predictions;
		# but it keeps the theano backend from crashing
		for tmp in ("recurrent_dropout", "dropout"):
			if tmp in rnn_config: 
				del rnn_config[tmp] 
		
		self.score_model.add(Bidirectional(self._WRAPPER(rnn.__class__(**rnn_config)), 
			merge_mode = "concat", weights = bidir.get_weights()))
		
class ScoreModelErasure(ScoreModelLinear):
	"""
	Omission, eq. 9
	"""
	_SCORE = "omission"
	
	def build_inner(self):
		if self.orig.architecture == "CNN":
			mainmodel = self.get_layer(Sequential)
			cnn = mainmodel.layers[0]
			main = Sequential([Conv1D(**cnn.get_config(), weights = cnn.get_weights()), GlobalMaxPooling1D()])
			
		else:
			bidir = self.get_layer(Bidirectional)
			rnn = bidir.forward_layer
			rnn_config = rnn.get_config()
			for tmp in ("recurrent_dropout", "dropout"):
				if tmp in rnn_config: del rnn_config[tmp]
			main = Bidirectional(rnn.__class__(**rnn_config), 
				merge_mode = "concat", weights = bidir.get_weights())
		self.score_model.add(ErasureWrapper(main))


class ScoreModelGradient(ScoreModel):
	"""
	Gradient score, eq. 6 & 7
	"""
	def __init__(self, orig, mode):
		ScoreModel.__init__(self, orig)
		self._SCORE = self._SCORE + mode
		self.mode = mode

	def build(self):
		pass

	def build_n(self, n):
		"""
		Since the Gradient Wrapper gets very slow when we ask it to calculate 20 classes at once, we build one model per target class.
		
		n: the index of the class that we are interested in
		"""
		print("Build", n)
		old_dense = self.get_layer(Dense)
		old_embedding = self.get_layer(Embedding)
		
		inp = Input((None,))
		self.score_model = Sequential([Embedding(**old_embedding.get_config(), weights = old_embedding.get_weights())])

		main = Sequential()
		input_shape = (None, old_embedding.output_dim)

		if self.orig.architecture == "CNN":
			mainmodel = self.get_layer(Sequential)
			old_cnn = mainmodel.layers[0]
			cnn = Conv1D(**old_cnn.get_config(), input_shape = input_shape, weights = old_cnn.get_weights())
			main.add(cnn)
			main.add(GlobalMaxPooling1D())
			
		else:
			old_bidir = self.get_layer(Bidirectional)
			rnn = old_bidir.forward_layer
			rnn_config = rnn.get_config()
			for tmp in ("recurrent_dropout", "dropout"):
				if tmp in rnn_config: 
					del rnn_config[tmp]
			bidir = Bidirectional(rnn.__class__(**rnn_config), input_shape=input_shape, 
				merge_mode = "concat", weights = old_bidir.get_weights())
			main.add(bidir)
		
		main.add(Dense(**old_dense.get_config(), weights = old_dense.get_weights()))
		main.add(Activation(self._ACTIVATION))

		self.score_model.add(GradientWrapper(main, mode = self.mode, out = n))
	
	def score_dataset(self, dataset, pred_only = False):
		self.orig.corpus.load_if_necessary("classdict")
		self.orig.corpus.load_if_necessary("X")
		self.score_model = 0
		if pred_only:
			return self.score_dataset_pred(dataset)
		return self.score_dataset_all(dataset)

	def score_dataset_pred(self, dataset):
		"""
		Calculate relevance scores only for the predicted classes
		"""
		X = self.orig.corpus.X[dataset]
		
		pred_generator = get_pred_generator(dataset, self.BATCH_SIZE, self.orig.architecture, self.orig.corpus)
		spe = self.orig.corpus.get_steps_per_epoch(dataset, self.BATCH_SIZE)
		
		pred = []
		for _ in range(spe):
			pred.extend(next(pred_generator))
		
		scores = [None for _ in range(len(X))]

		by_pred = {p:[] for p in range(len(self.orig.corpus.classdict))}
		for x, p, i in zip(X, pred, range(len(X))):
			# since we are going by target classes, we have to remember where samples originated
			by_pred[p].append((x, i))

		for p in by_pred:
			del self.score_model
			self.build_n(p)
			bar = ProgressBar()
			
			for x,i in bar(by_pred[p]):
				scores[i] = self.score_model.predict(np.array([x]))[0]

		assert all([not s is None for s in scores]) # make sure we have not left out anything
		return scores
			
				
		
	def score_dataset_all(self, dataset):
		"""
		Calculate relevance scores for all possible target classes
		"""
		stack = []
		for n in range(len(self.orig.corpus.classdict)):
			generator = self.orig.corpus.get_generator(dataset, self.BATCH_SIZE, shuffle = False)
			spe = self.orig.corpus.get_steps_per_epoch(dataset, self.BATCH_SIZE)
			
			del self.score_model
			self.build_n(n)
		
			bar = ProgressBar()

			scores = []

			for _ in bar(list(range(spe))):
				x, y = next(generator)
				score = self.score(x, None)
				for sample_x, sample_score in zip(x, score): 
					tmp = np.array([sample_score[i] for i in range(sample_x.shape[0]) if sample_x[i] != 0])
					scores.append(tmp)
			

			stack.append(scores)

		return [np.concatenate(x, axis = -1) for x in zip(*stack)]
		

		
class ScoreModelGradientRaw(ScoreModelGradient):
	_SCORE = "grad_raw"
	_ACTIVATION = "linear"


class ScoreModelGradientProb(ScoreModelGradient):
	_SCORE = "grad_prob"
	_ACTIVATION = "softmax"
		 
class ScoreModelBeta(ScoreModelBetaGamma):
	_WRAPPER = BetaDecomposition
	_SCORE = "beta"

class ScoreModelGamma(ScoreModelBetaGamma):
	_WRAPPER = GammaDecomposition
	_SCORE = "gamma"
	
				
class ScoreLime(Score):
	"""
	Subsequence LIME method, section 3.4
	"""
	_LIMEPARAMS = {"mode": "random", "nb_samples": 3000}
	def score(self, x, pred):
		tmp = self.lime.call(x, verbose = 0, out = pred)
		tmp = [np.array([tmp[i][j] for j in range(tmp.shape[1]) if x[i][j] != 0]) for i in range(tmp.shape[0])]
		return tmp

class ScoreLimeClass(ScoreLime):
	"""
	Black-box subsequence LIME, eq. 16
	"""
	_SCORE = "lime_class"

	def build(self):
		self.lime = TextLime(self.orig.model, loss = "binary_crossentropy", activation = "sigmoid", **self._LIMEPARAMS)

class ScoreLimeProb(ScoreLime):
	_SCORE = "lime_prob"
	
	def build(self):
		self.lime = TextLime(self.orig.model, loss = "mse", activation = "sigmoid", **self._LIMEPARAMS)

class ScoreLimeRaw(ScoreLime):
	"""
	Raw score subsequence LIME, eq. 17
	"""
	_SCORE = "lime_raw"
	
	def build(self):
		model_config = self.orig.model.get_config()
		model_config["layers"][-1]["config"]["activation"] = "linear"
		# copy the original architecture but replace the softmax activation with a linear activation
		copy = Model.from_config(model_config)
		copy.set_weights(self.orig.model.get_weights())
		self.lime = TextLime(copy, loss = "mse", activation = "linear", **self._LIMEPARAMS)



class ScoreLRP(Score):
	def build(self):
		"""
		Build relevance scoring models using third party code
		"""
		self.orig.corpus.load_if_necessary("classdict")
		if self.orig.architecture == "LSTM": # LRP_for_LSTM
			from LSTM_bidi import keras_to_weights, LSTM_bidi
			weights = keras_to_weights(self.orig.model, [0,2,1,3]) # get and refromat weights from original model
			self.score_model = LSTM_bidi(weights)
		elif self.orig.architecture == "GRU": # LRP_for_LSTM
			from LSTM_bidi import keras_to_weights, GRU_bidi
			weights = keras_to_weights(self.orig.model, [0,1,2])
			self.score_model = GRU_bidi(weights)
		else: 	# lrp_toolbox
			# the lrp toolbox does not support 1D convolution, global pooling or padding,
			# so we have to use some work-arounds

			from sequential import Sequential as LRPSequential
			from linear import Linear
			from maxpool import MaxPool
			from rect import Rect
			from convolution import Convolution
			from flatten import Flatten

			from keras.layers import ZeroPadding1D, Embedding
			
			conv = self.get_layer(Sequential).layers[0]
			dense = self.get_layer(Dense)
			emb = self.get_layer(Embedding)

			self.numclasses = len(self.orig.corpus.classdict)
			self.filterlength = conv.kernel_size[0]

			# self.embmodel outputs embeddings of size (batch_size, MAXLENGTH + 4, embedding_size)
			# the +4 are due to zeropadding (two to the left, two to the right, as our filterlength is 5)
			# in keras, the Conv1D module does the padding automaticalle
			self.embmodel = Sequential([Embedding(input_shape = (self.orig.corpus.MAXLENGTH,), 
				input_dim = emb.input_dim, output_dim = emb.output_dim, weights = emb.get_weights()),
				ZeroPadding1D(self.filterlength//2)])

			# create a lrp Linear layer with the same weights as our dense layer
			dense_lrp = Linear(conv.filters, dense.units)
			dense_lrp.W = dense.get_weights()[0] # set fully connected weights
			dense_lrp.B = dense.get_weights()[1] # set class biases
			
			# set pool length to MAXLENGTH in order to mimic the effect of global max pooling
			maxpool_lrp = MaxPool(pool=(self.orig.corpus.MAXLENGTH, 1))

			# create an lrp Convolution layer with the same weights as our conv layer
			# add an extra dimension to our convolution weights as the lrp toolbox only works with 2D inputs/filters
			# since the dimension is of size 1, it does nothing
			conv_lrp = Convolution(filtersize=(self.filterlength, emb.output_dim, 1, conv.filters), stride = (1,1))
			conv_lrp.W = np.expand_dims(conv.get_weights()[0], 2)
			conv_lrp.B = conv.get_weights()[1]

			# n.b.: the omission of the SoftMax layer is intentional
			# we propagate back raw scores in keeping with the code in the LRP_for_LSTM repo
			self.score_model = LRPSequential([conv_lrp, Rect(), maxpool_lrp, Flatten(), dense_lrp])

	def score(self, x, pred):
		"""
		Return relevance scores for x.
		
		pred: if none, return scores for all possible classes
		else, assume that pred is a list of target classes
		"""
		if self.orig.architecture == "CNN":
			return self.score_cnn(x, pred)
		else:
			return self.score_rnn(x, pred)
	
	def score_cnn(self, x, pred):
		"""
		Calculate relevance scores using the lrp toolbox.
		"""

		all_scores = []
		if pred is None: # if pred is None, assume that we want relevance scores for all possible classes
			pred = [list(range(len(self.orig.corpus.classdict))) for _ in range(x.shape[0])]
		else:
			pred = [[p] for p in pred]

		# pad all inputs to size MAXLENGTH
		# (remember that we have to commit to MAXLENGTH in order to make global max pooling work)
		xpadded = np.array([np.concatenate([x_s, np.zeros(self.orig.corpus.MAXLENGTH - x_s.shape[0])], axis = 0) for x_s in x])
		
		# get padded embeddings from the embedding model
		e = self.embmodel.predict_on_batch(xpadded)
		# add an extra dimension because the toolbox thinks we're doing 2D convolution
		e = np.expand_dims(e, axis = -1)
		
		for x_sample, e_sample, pred_sample in zip(x, e, pred):
			# we're doing batches of size 1
			scores = []
			e_sample = np.expand_dims(e_sample, 0) # reinsert the batch dimension
			for cl in pred_sample:
				forward = self.score_model.forward(e_sample)
				mask = np.zeros_like(forward)
				mask[:,cl] = 1 # mask all relevance scores except for the target class
				# in the RNN LRP code, this is done by the LRP module, so we do it here
				lrp = self.score_model.lrp(forward * mask, "epsilon", param = 0.001)

				# get rid of extra dimensions (batch size, 2D dimension)
				lrp = lrp.squeeze().sum(-1)

				# get rid of relevance scores for the +4 padding
				lrp = lrp[self.filterlength//2:-(self.filterlength//2)]

				# get rid of relevance scores for any other padding material that was added to the right
				lrp = np.array([lrp[i] for i in range(x_sample.shape[0])])
				scores.append(lrp)

			# stack relevance scores for all classes (only one class if pred_only)
			scores = np.stack(scores, -1)
			assert scores.shape == (x_sample.shape[0], len(pred_sample))
			all_scores.append(scores)
		return all_scores

	def score_rnn(self, x, pred):
		all_scores = []

		if pred is None: # if pred is None, assume that we want relevance scores for all possible classes
			pred = [list(range(len(self.orig.corpus.classdict))) for _ in range(x.shape[0])]
		else:
			pred = [[p] for p in pred]

		for x_sample, pred_sample in zip(x, pred):
			scores = []
			for cl in pred_sample:
				forward, backward, _ = self.score_model.lrp(x_sample.astype(int), cl, eps = 0.001, bias_factor = 0.0)
				# sum relevance scores from the forward and backward RNNs, then sum over the embedding dimension
				scores.append(np.sum(forward + backward, axis = -1))
			scores = np.stack(scores, -1)
			assert scores.shape == (x_sample.shape[0], len(pred_sample))
			all_scores.append(scores)
		return all_scores
		
		
