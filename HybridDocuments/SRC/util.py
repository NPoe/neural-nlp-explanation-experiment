from config import *
import os
import sys
import re
# add third party repos to pythonpath

L2SCORES = ["grad_raw_l2", "grad_prob_l2"]
L2SCORES += [s + "_integrated" for s in L2SCORES]
DOTSCORES = [re.sub("l2", "dot", s) for s in L2SCORES]
OMITSCORES = ["omit-1", "omit-3", "omit-7"]
LIMESCORES = ["limsse_" + s for s in ("class", "raw", "prob")]

OCCSCORES = [re.sub("omit", "occ", s) for s in OMITSCORES]

SCORES = L2SCORES + DOTSCORES + OMITSCORES + OCCSCORES + ["decomp", "lrp", "deeplift"] + LIMESCORES

CORPORA = ("newsgroup", "yelp")

ARCHITECTURES = ("GRU", "LSTM", "CNN", "QLSTM", "QGRU")

DIRECTORIES = {name: os.path.join(WORKDIR, name) for name in ("Models", "CSV", "Inputs", "Scores", "Predictions")}

JSONS = [TRAINJSON, DEVJSON, TESTJSON]

def prep_directories():
	"""
	Create all necessary directories
	"""
	for directory in DIRECTORIES.values():
		if not os.path.exists(directory):
			os.mkdir(directory)

	for corpus in CORPORA:
		tmp = os.path.join(DIRECTORIES["Inputs"], corpus)
		if not os.path.exists(tmp):
			os.mkdir(tmp)


def command_line_overlap(primary_list):
	"""
	Returns the list of all candidates from primary_list that are in the argv;
	important for scoring/training/evaluating more than one corpus/architecture/relevance score
	"""
	if "all" in sys.argv and "gamma" in primary_list:
		return primary_list
	return [x for x in primary_list if x in sys.argv]



		
## Utity functions that return paths of interest ##
###################################################

def make_modelpath(architecture, corpus):
	return os.path.join(DIRECTORIES["Models"], "_".join((architecture, corpus)) + ".hdf5")

def make_scorepath(dataset, architecture, corpus, score, pred_only):
	suffix = ""
	if pred_only: suffix = "_k"
	return os.path.join(DIRECTORIES["Scores"], "_".join((dataset, architecture, corpus, score)) + ".score" + suffix)

def make_csvpath(architecture, corpus):
	return os.path.join(DIRECTORIES["CSV"], "_".join((architecture, corpus)) + ".csv")

def make_predpath(dataset, architecture, corpus):
	return os.path.join(DIRECTORIES["Predictions"], "_".join((dataset, architecture, corpus)) + ".pred")

def make_storagedir(corpus):
	return os.path.join(DIRECTORIES["Inputs"], corpus)


prep_directories()
