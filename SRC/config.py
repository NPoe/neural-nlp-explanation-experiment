"""
Paths of necessary inputs, change if necessary!
"""

# directory where models, results etc. can be stored
WORKDIR = ".."

# text format pre-trained embedding
# format per line:
# word dim1 dim2 ... dim300
# any 300 dimensional embedding is fine, we use GloVe
GLOVEPATH = "../../Hybrid/Glove/glove.840B.300d.txt"

# json files containing sentiment analysis review data
# format per line:
# {'text': '...', 'stars': [1-5], ...}
TRAINJSON = "../../Hybrid/Data/reviews.PA.train"
DEVJSON = "../../Hybrid/Data/reviews.PA.dev"
TESTJSON = "../../Hybrid/Data/reviews.PA.test"

# locations of third party repositories
LRP_RNN_REPO = "../ThirdParty/LRP_for_LSTM/code/LSTM"
LRP_TOOLBOX_REPO = "../ThirdParty/lrp_toolbox/python/modules"

#####################################################################
