# download the manual evaluation benchmark [1]
git clone https://github.com/SinaMohseni/ML-Interpretability-Evaluation-Benchmark ML-Interpretability-Evaluation-Benchmark

cd ML-Interpretability-Evaluation-Benchmark/Text/org_documents/20news-bydate/20news-bydate-test/sci.electronics

# convert files to utf-8 if necessary
for F in *; do
  if file $F | grep 'ISO-8859'; then
    iconv -f ISO-8859-1 -t UTF-8 $F > tmp.txt
    mv tmp.txt $F
  fi
done

# [1] Mohseni, S., Ragan, E.D. (2018). A Human-Grounded Evaluation Benchmark for Local Explanations of Machine Learning. arXiv preprint arXiv:1801.05075.
