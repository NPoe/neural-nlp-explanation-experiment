cd Models

for model in GRU QGRU LSTM QLSTM CNN; do
  for corpus in yelp newsgroup; do
    wget "www.cis.uni-muenchen.de/~poerner/blobs/neural_nlp_explanation_experiment/${model}_${corpus}.hdf5";
  done;
done; 
