import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from scipy.sparse import issparse, lil_matrix, diags


class PerDocumentMaxNormalizer(BaseEstimator, TransformerMixin):
  """Normalizes each row by its own max value, producing values in [0, 1]."""

  def fit(self, X, y=None):
    self.is_fitted_ = True
    return self

  def transform(self, X):
    if not issparse(X):
      X = lil_matrix(X)
    X = X.tocsr().astype(float)
    row_maxes = np.array(X.max(axis=1).todense()).flatten()
    row_maxes[row_maxes == 0] = 1
    return diags(1.0 / row_maxes) @ X


class PositionalDistributionTransformer(BaseEstimator, TransformerMixin):
  """
  For each vocabulary term, produces 3 features (beginning, middle, end ratio)
  describing what proportion of that term's occurrences fall in each third
  of the document. Position is determined by the first token index of the ngram.

  Documents with fewer than 3 tokens place everything in the end bin.
  """

  def __init__(self, ngram_range=(1, 1)):
    self.ngram_range = ngram_range

  def _generate_ngrams(self, tokens):
    ngrams = []
    for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
      for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i: i + n])
        ngrams.append((ngram, i))
    return ngrams

  def _get_bin(self, position, doc_length):
    if doc_length < 3:
      return 2  # end
    third = doc_length / 3
    if position < third:
      return 0  # beginning
    elif position < 2 * third:
      return 1  # middle
    else:
      return 2  # end

  def fit(self, X, y=None):
    vocab = set()
    for doc in X:
      tokens = doc.split()
      for ngram, _ in self._generate_ngrams(tokens):
        vocab.add(ngram)
    self.vocabulary_ = sorted(vocab)
    self.vocab_index_ = {term: i for i, term in enumerate(self.vocabulary_)}
    return self

  def transform(self, X):
    n_vocab = len(self.vocabulary_)
    result = lil_matrix((len(X), n_vocab * 3))

    for doc_idx, doc in enumerate(X):
      tokens = doc.split()
      doc_length = len(tokens)
      ngrams = self._generate_ngrams(tokens)

      term_bin_counts = {}
      term_total_counts = {}

      for ngram, position in ngrams:
        if ngram not in self.vocab_index_:
          continue
        term_idx = self.vocab_index_[ngram]
        bin_idx = self._get_bin(position, doc_length)

        key = (term_idx, bin_idx)
        term_bin_counts[key] = term_bin_counts.get(key, 0) + 1
        term_total_counts[term_idx] = term_total_counts.get(term_idx, 0) + 1

      for (term_idx, bin_idx), count in term_bin_counts.items():
        total = term_total_counts[term_idx]
        result[doc_idx, term_idx * 3 + bin_idx] = count / total

    return result.tocsr()


class LogisticRegressionModel:
  def __init__(
      self, random_state=67, ngram_range=(1, 1), use_positional_features=False
  ):
    self.random_state = random_state
    self.ngram_range = ngram_range
    self.use_positional_features = use_positional_features

    if self.use_positional_features:
      self.model = Pipeline(
          [
              (
                  "features",
                  FeatureUnion(
                      [
                          (
                              "counts",
                              Pipeline(
                                  [
                                      (
                                          "vectorizer",
                                          CountVectorizer(
                                              ngram_range=self.ngram_range
                                          ),
                                      ),
                                      ("normalizer", PerDocumentMaxNormalizer()),
                                  ]
                              ),
                          ),
                          (
                              "positional",
                              PositionalDistributionTransformer(
                                  ngram_range=self.ngram_range
                              ),
                          ),
                      ]
                  ),
              ),
              (
                  "clf",
                  LogisticRegression(
                      solver="lbfgs",
                      max_iter=1000,
                      random_state=self.random_state,
                  ),
              ),
          ]
      )
    else:
      self.model = Pipeline(
          [
              ("vectorizer", CountVectorizer(ngram_range=self.ngram_range)),
              (
                  "clf",
                  LogisticRegression(
                      solver="lbfgs",
                      max_iter=1000,
                      random_state=self.random_state,
                  ),
              ),
          ]
      )

  def fit(self, training_set):
    X_train = [" ".join(conversation["text"]) for conversation in training_set]
    Y_train = [conversation["manipulation_type"]
               for conversation in training_set]
    self.model.fit(X_train, Y_train)

  def predict(self, test_set, output_as_dict=False):
    X_test = [" ".join(conversation["text"]) for conversation in test_set]
    Y_test = [conversation["manipulation_type"] for conversation in test_set]
    Y_pred = self.model.predict(X_test)
    return classification_report(Y_test, Y_pred, zero_division=0, output_dict=output_as_dict)
