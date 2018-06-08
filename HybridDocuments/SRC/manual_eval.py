from util import *
import numpy as np
np.random.seed(12345)
import _pickle
import json
import sys
from util import *
from corpora import *
from task_methods import *
from explanation_methods import *
from nltk.tokenize import word_tokenize

from progressbar import ProgressBar

class ManualEval:
	def __init__(self, architecture):
		self.architecture = architecture

	def build(self):
		self.corpus = get_corpus_object("newsgroup")
		self.corpus.load_select_if_necessary(["classdict", "worddict"])
		self.load_gt()
		self.model = TaskMethod(self.architecture, self.corpus)
		self.model.load(make_modelpath(self.architecture, "newsgroup"))

	def build_score_model(self, score):
		if score != "random":
			self.score_model = self.model.make_explanation_method(score)
			self.score_model.build()
		
	def eval(self, score):
		self.build_score_model(score)
		hits = 0
		total = 0
		
		if score == "random":
			np.random.seed(12345)

		for cl in self.CLASSES:
			cl = self.corpus.classdict[cl]
			if "grad" in score:
				self.score_model.build_n(cl)
			
			bar = ProgressBar()
			for x,g,raw in bar(list(zip(self.X[cl], self.GT[cl],self.raw[cl]))):
				if self.model.model.predict(np.expand_dims(x,0))[0].argmax() == cl:
					peak = self.point(x, score, cl,g,raw,total)
					hits += int(peak in g)
					total += 1
						

		print(self.architecture, score)
		print(hits, "/", total, "=", hits/total)
		print(flush=True)

	def point(self, x, score, cl, g, raw, i):
		if score == "random":
			return np.random.randint(len(x))
		else:
			s = self.score_model.score(np.expand_dims(x, 0), pred = np.array([cl]))[0].squeeze()
			assert(s.shape == x.shape)
			return s.argmax()
				
			
	def load_gt(self):
		ORIGPATH = os.path.join(MANUAL_BENCHMARK, "Text/org_documents/20news-bydate/20news-bydate-test/")
		EVALPATH = os.path.join(MANUAL_BENCHMARK, "Text/user_evaluation/")

		self.CLASSES = os.listdir(ORIGPATH)

		self.X = {}
		self.GT = {}
		self.raw = {}

		lengths = []
		for cl in self.CLASSES:
			clname = self.corpus.classdict[cl]
			self.X[clname] = []
			self.GT[clname] = []
			self.raw[clname] = []
			for doc in os.listdir(os.path.join(ORIGPATH, cl)):
				with open(os.path.join(ORIGPATH, cl, doc)) as handle:	
					tmp = handle.read().strip()
					if len(tmp):
						tokenized = word_tokenize(tmp)[:1000]
						tokenized_lk = [w.lower() for w in word_tokenize(tmp)]
						jpath = os.path.join(EVALPATH, cl, doc + ".json")
						if os.path.exists(jpath):	
							self.X[clname].append(np.array([self.corpus.worddict.get(w, 1) for w in tokenized]))
							self.raw[clname].append(tokenized)
							with open(os.path.join(EVALPATH, cl, doc + ".json")) as jhandle:
								jobj = json.load(jhandle)
								words = [w[0] for w in jobj["words"]]
								lengths.append(len(word_tokenize(tmp)))
								for word in words:
									if not word in tokenized_lk:
										a = [w for w in tokenized_lk if word in w]
								indices = [i for i, x in enumerate(tokenized_lk) if any([x.startswith(_) or x.endswith(_) for _ in words])]
								self.GT[clname].append(np.array(indices))
		
