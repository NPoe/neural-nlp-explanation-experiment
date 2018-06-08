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
		return self.score_from_generator(generator, spe, pred_generator = pred_generator, dataset = dataset)
	
	def score_from_generator(self, generator, spe, pred_generator, dataset):
		"""
		Produce relevance scores from a generator

		generator: generator that returns (X,Y) tuples where X has the shape (batch_size, num_words)
		spe: steps per epoch
		pred_generator: generator returning arrays of target classes of the shape (batch_size,); can be None
		"""
		bar = ProgressBar()

		scores = []

		for _ in bar(list(range(spe))):
			x, y = next(generator)

			if pred_generator is None:
				pred = None
			else:
				pred = next(pred_generator)
			
			scores.extend(self.score(x, pred))
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

		#print("Score model passed all checks")
		
		#self.score_model.summary()

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
	Decomposition scores (Murdoch & Szlam 2017), section 3.4
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
	Omission or occlusion, section 3.5
	"""
	
	def __init__(self, orig, mode, n_gram = 1, **kwargs):
		super(ScoreModelErasure, self).__init__(orig, **kwargs)
		self.n_gram = n_gram
		self.mode = mode

	def build_inner(self):
		if self.orig.architecture == "CNN":
			mainmodel = self.get_layer(Sequential)
			cnns = [l for l in mainmodel.layers if isinstance(l, Conv1D)]
			main = Sequential([Conv1D(**cnn.get_config(), weights = cnn.get_weights()) for cnn in cnns])
			main.add(GlobalMaxPooling1D())
		else:
			bidir = self.get_layer(Bidirectional)
			rnn = bidir.forward_layer
			rnn_config = rnn.get_config()
			for tmp in ("recurrent_dropout", "dropout"):
				if tmp in rnn_config: del rnn_config[tmp]
			main = Bidirectional(rnn.__class__(**rnn_config), 
				merge_mode = "concat", weights = bidir.get_weights())
		self.score_model.add(ErasureWrapper(main, ngram = self.n_gram, mode = self.mode))
	
	def score(self, x, pred):
		x = np.concatenate([x, np.zeros((x.shape[0], max(0, self.n_gram-x.shape[1])))], axis=1)
		orig = self.score_model.predict(x)
		if not pred is None:
			orig = np.array([t[:,p:p+1] for t,p in zip(orig, pred)])

		tmp = []
		for o in orig:
			stack = []
			for n in range(self.n_gram):
				left = np.zeros((self.n_gram - 1 - n, o.shape[1]))
				right = np.zeros((n, o.shape[1]))
				stack.append(np.concatenate([left, o, right], axis = 0))
			stack = np.stack(stack, axis = 0)
			mean = np.sum(stack, axis = 0) / self.n_gram	
			assert mean.shape == (o.shape[0] + self.n_gram - 1, o.shape[1])
			tmp.append(mean)
		tmp = np.array(tmp)
		tmp = [np.array([tmp[i][j] for j in range(tmp.shape[1]) if x[i][j] != 0]) for i in range(tmp.shape[0])]
		return tmp


class ScoreModelGradient(ScoreModel):
	"""
	Gradient scores, section 3.1
	"""
	def __init__(self, orig, mode, integrated = False, **kwargs):
		ScoreModel.__init__(self, orig, **kwargs)
		self._SCORE = self._SCORE + mode
		self.mode = mode
		# simple gradient means integrated gradient with M = 1
		self.num_alpha = 1 + 49 * int(integrated)

	def build(self):
		old_embedding = self.get_layer(Embedding)
		self.embmodel = Sequential([Embedding(**old_embedding.get_config(), weights = old_embedding.get_weights())])

	def build_n(self, n):
		"""
		Since the Gradient Wrapper gets very slow when we ask it to calculate 20 classes at once, we build one model per target class.
		
		n: the index of the class that we are interested in
		"""
		old_dense = self.get_layer(Dense)
		old_embedding = self.get_layer(Embedding)
		
		inp = Input((None,))
		self.score_model = Sequential()#[old_embedding])

		main = Sequential()
		input_shape = (None, old_embedding.output_dim)

		if self.orig.architecture == "CNN":
			mainmodel = self.get_layer(Sequential)
			cnns = [l for l in mainmodel.layers if isinstance(l, Conv1D)]
			cnns = [Conv1D(**cnn.get_config(), weights = cnn.get_weights()) for cnn in cnns]
			for cnn in cnns:
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

		self.score_model.add(GradientWrapper(main, mode = None, out = n, input_shape = input_shape, num_alpha = 1))
	
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
				E = self.embmodel.predict(np.array([x[:self.orig.corpus.MAXLENGTH]]))

				tmp = 0
				for j in range(self.num_alpha):
					tmp += self.score_model.predict(E * (j+1) / self.num_alpha)[0]
				tmp /= self.num_alpha
				
				if self.mode == "dot":
					scores[i] = np.sum(np.expand_dims(E[0], -1) * tmp, axis = 1)
				elif self.mode == "l2":
					scores[i] = np.sqrt(np.sum(tmp * tmp, axis = 1))

				assert len(scores[i].shape) == 2 and scores[i].shape[0] == x[:self.orig.corpus.MAXLENGTH].shape[0]

		assert all([not s is None for s in scores]) # make sure we have not left out anything
		return scores
			
	def score(self, x, pred=None):
		E = self.embmodel.predict(x[:,:self.orig.corpus.MAXLENGTH])

		tmp = 0
		for j in range(self.num_alpha):
			tmp += self.score_model.predict(E * (j+1) / self.num_alpha)
		tmp /= self.num_alpha
				
		if self.mode == "dot":
			return np.sum(np.expand_dims(E, -1) * tmp, axis = 2)
		elif self.mode == "l2":
			return np.sqrt(np.sum(tmp * tmp, axis = 2))
			
		
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
	def __init__(self, orig, mode, integrated = False, **kwargs):
		super(ScoreModelGradientProb, self).__init__(orig = orig, mode = mode, integrated = integrated, **kwargs)
		self._ACTIVATION = "softmax"
		 
class ScoreModelBeta(ScoreModelBetaGamma):
	_WRAPPER = BetaDecomposition
	_SCORE = "beta"

class ScoreModelGamma(ScoreModelBetaGamma):
	_WRAPPER = GammaDecomposition
	_SCORE = "gamma"
	
				
class ScoreLime(Score):
	"""
	LIMSSE method, section 3.6
	"""
	def score(self, x, pred):
		if not pred is None:
			tmp = self.lime.call(x, verbose = 0, out = pred)
		else:
			tmp = self.lime.call(x, verbose = 0)
		tmp = [np.array([tmp[i][j] for j in range(tmp.shape[1]) if x[i][j] != 0]) for i in range(tmp.shape[0])]
		return tmp

	def build(self):
		self.lime = self._LIMECLASS(self.orig.model, nb_samples = 3000, **self._LIMEPARAMS)

class ScoreLimeClass(ScoreLime):
	_SCORE = "lime_class"
	_LIMECLASS = TextLime
	_LIMEPARAMS = {"loss": "binary_crossentropy", "activation": "sigmoid", "minlength": 1, "maxlength": 7, "mode": "random"}

class ScoreLimeProb(ScoreLime):
	_SCORE = "lime_prob"
	_LIMECLASS = TextLime
	_LIMEPARAMS = {"loss": "mse", "activation": "linear", "minlength": 1, "maxlength": 7, "mode": "random"}
	
class ScoreLimeRaw(ScoreLime):
	_SCORE = "lime_raw"
	_LIMEPARAMS = {"loss": "mse", "activation": "linear", "minlength": 1, "maxlength": 7, "mode": "random"}
	
	def build(self):
		model_config = self.orig.model.get_config()
		model_config["layers"][-1]["config"]["activation"] = "linear"
		# copy the original architecture but replace the softmax activation with a linear activation
		copy = Model.from_config(model_config)
		copy.set_weights(self.orig.model.get_weights())
		self.lime = TextLime(copy, **self._LIMEPARAMS)


class ScoreThirdParty(Score):
	def build(self):
		"""
		Build relevance scoring models using third party code
		"""
		sys.path.append(LRP_RNN_REPO)
		if self.orig.architecture == "LSTM": # LRP_for_LSTM
			from layers import keras_to_weights_birnn, BILSTM_np
			weights = keras_to_weights_birnn(self.orig.model, [0,2,1,3]) # get and refromat weights from original model
			self.score_model = BILSTM_np(weights, mode = self.MODE)
		elif self.orig.architecture == "GRU": # LRP_for_LSTM
			from layers import keras_to_weights_birnn, BIGRU_np
			weights = keras_to_weights_birnn(self.orig.model, [0,1,2])
			self.score_model = BIGRU_np(weights, mode = self.MODE)
		elif self.orig.architecture == "QLSTM": # LRP_for_LSTM
			from layers import keras_to_weights_biqrnn, BIQLSTM_np
			weights = keras_to_weights_biqrnn(self.orig.model, [0,2,1,3]) # get and refromat weights from original model
			self.score_model = BIQLSTM_np(weights, mode = self.MODE)
		elif self.orig.architecture == "QGRU": # LRP_for_LSTM
			from layers import keras_to_weights_biqrnn, BIQGRU_np
			weights = keras_to_weights_biqrnn(self.orig.model, [0,1])
			self.score_model = BIQGRU_np(weights, mode = self.MODE)
		elif self.orig.architecture == "CNN":
			from layers import keras_to_weights_cnn, CNN_np
			weights = keras_to_weights_cnn(self.orig.model)
			self.score_model = CNN_np(weights, mode = self.MODE)
		sys.path.pop()

	def score(self, x, pred):
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
	
class ScoreDL(ScoreThirdParty):
	"""DeepLIFT method, section 3.3"""
	MODE = "dl"

class ScoreLRP(ScoreThirdParty):
	"""LRP method, section 3.2"""
	MODE = "lrp"
