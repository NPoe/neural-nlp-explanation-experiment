from util import *
import numpy as np
np.random.seed(12345)
import _pickle

def get_pointing_game(score):
	"""
	Returns the correct pointing game contestant for a given score
	"""
	if score == "random": return NaiveRandomPointingGame
	if score == "biased": return BiasedRandomPointingGame
	return ScorePointingGame

class PointingGame:
	"""
	Pointing game contestant.
	"""
	def __init__(self, architecture, corpus, score, pred_only):
		self.architecture = architecture
		self.corpus = corpus
		self.score = score
		# if pred_only, expect relevance scores of the shape (num_words, 1); else (num_words, num_classes)
		self.pred_only = pred_only 
		self.points = {}
	
	def load(self, path):
		with open(path, "rb") as handle:
			tmp = _pickle.load(handle)
		return tmp
	
	def load_groundtruths(self):
		groundtruthpath = os.path.join(make_storagedir(self.corpus), "GT")
		self.groundtruths = self.load(groundtruthpath)
			
	def load_predictions(self):
		predpath = make_predpath("hybrid", self.architecture, self.corpus)
		self.predictions = self.load(predpath)

	def valid(self, i):
		"""
		Returns true if the task method predicted a class that is the gold label of at least one word
		"""
		return self.predictions[i] in self.groundtruths[i]

	def play(self):
		print("Pointing game")
		print("Architecture", self.architecture),
		print("Corpus", self.corpus)
		print("Score", self.score)

		hits, discarded = 0, 0
		total = len(self.predictions)

		for i in range(total):
			peak = self.point(i)

			if self.valid(i):
				hits += int(self.groundtruths[i][peak] == self.predictions[i])
			else:
				discarded += 1

		print("Discarded", discarded, "/", total, "=", discarded / total)
		print("Pointing game accuracy", hits, "/ (", total, "-", discarded, ") =", hits / (total-discarded))
		print()	

		return hits / (total-discarded)

class ScorePointingGame(PointingGame):
	"""
	Pointing game contestant that places relevance peak according to some relevance scoring method. 
	"""

	def point(self, i):
		if self.pred_only:
			return np.squeeze(self.scores[i]).argmax(0)
		
		prediction = self.predictions[i]
		return self.scores[i].argmax(0)[prediction]
	
	def load_scores(self):
		scorepath = make_scorepath("hybrid", self.architecture, self.corpus, self.score, self.pred_only)
		self.scores = self.load(scorepath)
	
	def prepare(self):
		self.load_groundtruths()
		self.load_predictions()
		self.load_scores()
		assert len(self.groundtruths) == len(self.predictions) == len(self.scores)
		

class RandomPointingGame(PointingGame):
	"""
	Parent class for random pointing game contestant.
	"""	
	def prepare(self):
		self.randstate = np.random.RandomState(123)
		self.load_groundtruths()
		self.load_predictions()
		assert len(self.groundtruths) == len(self.predictions)

		
class NaiveRandomPointingGame(RandomPointingGame):
	"""
	Pointing game contestant that randomly picks a relevance peak in the document.
	"""
	def point(self, i):
		length = self.groundtruths[i].shape[0]
		return self.randstate.randint(0, length)

class BiasedRandomPointingGame(RandomPointingGame):
	"""
	Random pointing game contestant that has information about probable locations of truly relevant words.
	Doesn't really outperform the naive baseline.
	"""
	def prepare(self):
		super(BiasedRandomPointingGame, self).prepare()
		positions_dict = {}
		total = 0

		for prediction, groundtruth in zip(self.predictions, self.groundtruths):
			for i in range(groundtruth.shape[0]):
				if groundtruth[i] == prediction:
					position = float(i) / groundtruth.shape[0]
					positions_dict[position] = positions_dict.get(position, 0) + 1
					total += 1
		
		self.positions = list(positions_dict.keys())		
		self.probabilities = [positions_dict[position] / float(total) for position in self.positions]

	def point(self, i):
		length = self.groundtruths[i].shape[0]
		r = self.randstate.choice(self.positions, p = self.probabilities)
		return int(r * length)
		

