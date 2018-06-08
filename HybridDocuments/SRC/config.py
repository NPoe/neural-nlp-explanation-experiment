"""
Paths of necessary inputs, change if necessary!
"""

# directory where models, results etc. can be stored
WORKDIR = ".."

# text format pre-trained embedding
# format per line:
# word dim1 dim2 ... dim300
# any 300 dimensional embedding is fine, we use GloVe
GLOVEPATH = "../Glove/glove.840B.300d.txt"

# json files containing sentiment analysis review data
# format per line:
# {'text': '...', 'stars': [1-5], ...}
TRAINJSON = "../Data/reviews.PA.train"
DEVJSON = "../Data/reviews.PA.dev"
TESTJSON = "../Data/reviews.PA.test"

# locations of third party repositories
LRP_RNN_REPO = "../ThirdParty/LRP_and_DeepLIFT/code"

# location of manual interpretability benchmark	
MANUAL_BENCHMARK = "../ML-Interpretability-Evaluation-Benchmark"

#####################################################################
