# SVD

An implementation of the greedy algorithm for SVD, using the power method for the 1-dimensional case.

For the post [Singular Value Decomposition Part 2: Theorem, Proof, Algorithm](http://jeremykun.com/2016/05/16/singular-value-decomposition-part-2-theorem-proof-algorithm/)

And the first (motivational) post in the series: [Singular Value Decomposition Part 1: Perspectives on Linear Algebra](https://jeremykun.com/2016/04/18/singular-value-decomposition-part-1-perspectives-on-linear-algebra/)

# Setup

Run the following to set up all the requirements needed to run the code in this repository.

```
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ bash setup.sh   # downloads relevant NLP corpora from nltk
```

Then run `python3 topicmodel.py` for the main topic-model routine, `svd.py` for the core svd algorithm, and `demo.py` for the numpy examples from the post.

When finished, run `$ deactivate` to exit the virtual environment.
