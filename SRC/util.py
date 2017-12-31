from config import *
import os
import sys

# add third party repos to pythonpath
sys.path.append(LRP_RNN_REPO)
sys.path.append(LRP_TOOLBOX_REPO)

SCORES = ("gamma", "lrp", "omission", "lime_raw", "lime_class", "grad_prob_dot", "grad_raw_dot", 
	"grad_prob_l2", "grad_raw_l2")

CORPORA = ("newsgroup", "yelp")

ARCHITECTURES = ("GRU", "LSTM", "CNN")

DIRECTORIES = {name: os.path.join(WORKDIR, name) for name in ("Models", "CSV", "Inputs", "Scores", "Predictions", "Deletion")}

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
	return [x for x in primary_list if x in sys.argv]


		
## Utity functions that return paths of interest ##
###################################################
	
def make_deletionpath(architecture, corpus):
	return os.path.join(DIRECTORIES["Deletion"], "_".join((architecture, corpus)) + ".pickle")

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
