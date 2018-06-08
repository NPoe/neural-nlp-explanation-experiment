# Code for hybrid document experiment in [1].

## CORPORA

The 20 newsgroups corpus is available from sklearn and will be downloaded when running 

```main.py prepare newsgroup```

The manual evaluation benchmark [2] can be downloaded by running 

```./prep_manual```

The 10th yelp dataset challenge is no longer available on-line, and we do not have the license to publish it. 
However, the majority of relevant reviews (203756 out of 206338) are present in the [11'th
dataset challenge](https://www.yelp.com/dataset/download), and hopefully also in future editions. Note that
we only used reviews from the Pennsylvania area (state = 'PA'). The file [reviews.json](reviews.json) contains the IDs of all reviews that we used, along with our train/dev/test split.

## MODELS

Prior to training, you will need to download pre-trained embeddings [here](http://nlp.stanford.edu/data/glove.840B.300d.zip).
Then set this variable in the [config file](SRC/config.py):

```GLOVEPATH = # path to embedding txt file```

Then run

```
cd SRC
main.py prepare corpus
main.py train corpus architecture
```

where

```
# corpus is one of yelp|newsgroup
# architecture is one of CNN|GRU|LSTM|QGRU|QLSTM  
```

For reproducibility, you may want to skip the second step and instead download the models from our original experiment:

```./prep_models.sh```

## RUNNING THE EXPERIMENT

```
main.py eval architecture corpus # evaluate primary model performance on test set
main.py score architecture corpus method # pre-calculate relevance scores
main.py pointinggame architecture corpus method # evaluate relevance maximum
main.py manual architecture> method # evaluate relevance maximum on manual benchmark
```

where

```
# method is one of limsse_raw|limsse_class|omit-1|occ-1|grad_raw_dot| ... 
```

See [SRC/util.py](SRC/util.py) for full list of explanation methods.

## KNOWN ISSUES

There used to be a groundtruth-prediction mismatch on trimmed documents (i.e., documents with a length > 1000 words). 
This means that results reported in [1] underestimate pointing game accuracy on very long documents. This bug has 
been fixed in the present codebase.

## REFERENCES

[1] Poerner, N., Roth, B., Schütze, H. (2018). Evaluating neural network explanation methods using hybrid
documents and morphosyntactic agreement. ACL.

[2] Mohseni, S., Ragan, E.D. (2018) A Human-Grounded Evaluation Benchmark for Local Explanations of Machine Learning. 
arXiv preprint arXiv:1801.05075
