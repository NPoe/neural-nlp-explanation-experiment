from util import *
from corpora import *
from task_methods import *
from explanation_methods import *
from manual_eval import *
from pointing_game import *
import _pickle

def evaluate(architecture, corpus):
	"""
	Print test and dev set accuracy and loss of <architecture> on <corpus>
	"""
	corpus_object = get_corpus_object(corpus)
	model = TaskMethod(architecture, corpus_object)
	model.load(make_modelpath(architecture, corpus))
	model.evaluate()

def train(architecture, corpus):
	"""
	Train <architecture> on <corpus>
	"""
	corpus_object = get_corpus_object(corpus)
	
	model = TaskMethod(architecture, corpus_object)
	model.build()


	model.train(make_modelpath(architecture, corpus), make_csvpath(architecture, corpus))

 
def predict(architecture, corpus):
	"""
	Let <architecture> predict classes for all test set and hybrid documents in <corpus.

	Store predictions for later use.
	"""
	
	corpus_object = get_corpus_object(corpus)
	
	model = TaskMethod(architecture, corpus_object)
	
	model.load(make_modelpath(architecture, corpus))
	
	tmp = model.predict_dataset("hybrid")
	with open(make_predpath("hybrid", architecture, corpus), "wb") as handle:
		_pickle.dump(tmp, handle)		


def calculate_score(architecture, corpus, score, pred_only = False):
	"""
	Let <score> calculate relevance scores for <architecture> for all test set and hybrid documents in <corpus>
	
	pred_only: if true, calculate only relevance scores for the classes predicted by <architecture>
	else, calculate relevance scores for all possible target classes
	"""

	corpus_object = get_corpus_object(corpus)

	model = TaskMethod(architecture, corpus_object)
	modelpath = make_modelpath(architecture, corpus)

	model.load(make_modelpath(architecture, corpus))

	score_model = model.make_explanation_method(score)
	score_model.build()

	for dataset in ("hybrid",):# "test"):
		tmp = score_model.score_dataset(dataset, pred_only = pred_only)

		with open(make_scorepath(dataset, architecture, corpus, score, pred_only), "wb") as handle:
			_pickle.dump(tmp, handle)


def prepare(corpus):
	"""
	Prepare all input files for <corpus>
	"""
	corpus_object = get_corpus_object(corpus)
	corpus_object.prepare()

def datatest(corpus):
	"""
	Do some sanity checks on input files for <corpus>
	"""
	corpus_object = get_corpus_object(corpus)
	corpus_object.sanity_check()

	

if __name__ == "__main__":
	
	if "prepare" in sys.argv:	
		for corpus in command_line_overlap(CORPORA):
			prepare(corpus)

	if "datatest" in sys.argv:
		for corpus in command_line_overlap(CORPORA):
			datatest(corpus)
	
	if "train" in sys.argv:
		for corpus in command_line_overlap(CORPORA):
			for architecture in command_line_overlap(ARCHITECTURES):
				train(architecture, corpus)
	
	if "eval" in sys.argv:
		for corpus in command_line_overlap(CORPORA):
			for architecture in command_line_overlap(ARCHITECTURES):
				evaluate(architecture, corpus) 
	
	if "predict" in sys.argv:
		for corpus in command_line_overlap(CORPORA):
			for architecture in command_line_overlap(ARCHITECTURES):
				predict(architecture, corpus)

	if "score" in sys.argv:
		for corpus in command_line_overlap(CORPORA):
			for architecture in command_line_overlap(ARCHITECTURES):
				for score in command_line_overlap(SCORES):
					calculate_score(architecture, corpus, score, pred_only = True)
	
	if "pointinggame" in sys.argv:
		for corpus in command_line_overlap(CORPORA):
			for architecture in command_line_overlap(ARCHITECTURES):
				for score in command_line_overlap(SCORES) + ["random"]:
					game = get_pointing_game(score)(architecture, corpus, score, pred_only = True)
					game.prepare()
					game.play()
	if "manual" in sys.argv:
		for architecture in command_line_overlap(ARCHITECTURES):
			game = ManualEval(architecture)
			game.build()
			for score in command_line_overlap(SCORES) + ["random"]:
				if architecture == "CNN" and score == "decomp": continue
				game.eval(score)	
