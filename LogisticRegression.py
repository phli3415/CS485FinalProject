import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from scipy.sparse import issparse, lil_matrix, diags
from collections import Counter
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler



_YOU_TOKENS = {
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'youd', 'youll', 'youre', 'youve',
}
_I_TOKENS = {
    'i', 'me', 'my', 'mine', 'myself',
    'id', 'im', 'ive',
}
_NEG_TOKENS = {
    'no', 'nor', 'not', 'never', 'none', 'nothing', 'nobody', 'without',
    'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'cant', 'couldnt',
    'shouldnt', 'isnt', 'wasnt', 'werent', 'havent', 'hasnt', 'hadnt',
    'aint', 'mightnt', 'mustnt', 'neednt', 'shant',
}

_BE_FORMS = {
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    "'m", "'re", "'s",
}

DENSE_FEATURE_NAMES = [
    'md_rate',
    'prp_rate',
    'prp_poss_rate',
    'wh_rate',
    'sup_rate',
    'comp_rate',
    'vbd_rate',
    'pres_rate',
    'vbg_rate',
    'uh_rate',
    'mean_sent_len',
    'you_rate',
    'i_rate',
    'neg_rate',
    'rb_rate',
    'jj_rate',
    'passive_rate',
]


def _count_passive(word_pos_pairs, window=3):
  n = len(word_pos_pairs)
  if n == 0:
    return 0

  count = 0
  i = 0
  while i < n:
    tok, _ = word_pos_pairs[i]
    if tok.lower() in _BE_FORMS:
      for j in range(i + 1, min(i + 1 + window, n)):
        _, tag_j = word_pos_pairs[j]
        if tag_j == 'VBN':
          count += 1
          i = j
          break
        if tag_j in ('VBZ', 'VBD', 'VBP', 'MD'):
          break
    i += 1
  return count


def _compute_dense_features(text_tokens, pos_tags, word_pos_pairs=None):
  pos_tags = pos_tags or []
  text_tokens = text_tokens or []
  word_pos_pairs = word_pos_pairs or []

  n_pos = len(pos_tags)
  pos_denom = max(n_pos, 1)
  text_denom = max(len(text_tokens), 1)

  tc = Counter(pos_tags)

  md       = tc.get('MD', 0)
  prp      = tc.get('PRP', 0)
  prp_poss = tc.get('PRP$', 0)
  wh       = tc.get('WP', 0) + tc.get('WDT', 0) + tc.get('WRB', 0) + tc.get('WP$', 0)
  sup      = tc.get('JJS', 0) + tc.get('RBS', 0)
  comp     = tc.get('JJR', 0) + tc.get('RBR', 0)
  vbd      = tc.get('VBD', 0)
  pres     = tc.get('VBP', 0) + tc.get('VBZ', 0)
  vbg      = tc.get('VBG', 0)
  uh       = tc.get('UH', 0)
  rb       = tc.get('RB', 0)
  jj       = tc.get('JJ', 0)

  num_sents = max(tc.get('.', 0), 1)
  mean_sent_len = n_pos / num_sents

  you_count = sum(1 for t in text_tokens if t in _YOU_TOKENS)
  i_count   = sum(1 for t in text_tokens if t in _I_TOKENS)
  neg_count = sum(1 for t in text_tokens if t in _NEG_TOKENS)

  passive = _count_passive(word_pos_pairs)
  passive_denom = max(len(word_pos_pairs), 1)

  return np.array([
    md       / pos_denom,
    prp      / pos_denom,
    prp_poss / pos_denom,
    wh       / pos_denom,
    sup      / pos_denom,
    comp     / pos_denom,
    vbd      / pos_denom,
    pres     / pos_denom,
    vbg      / pos_denom,
    uh       / pos_denom,
    mean_sent_len,
    you_count / text_denom,
    i_count   / text_denom,
    neg_count / text_denom,
    rb       / pos_denom,
    jj       / pos_denom,
    passive  / passive_denom,
  ], dtype=np.float64)



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
  """Logistic regression with three optional feature blocks:
  word ngrams (always on), positional distribution, dense POS.
  Use class_weight='balanced' for class imbalance handling."""
  def __init__(
      self, 
      random_state=67, 
      ngram_range=(1, 1), 
      use_positional_features=False,
      use_pos_features=False,
  ):
    self.random_state = random_state
    self.ngram_range = ngram_range
    self.use_positional_features = use_positional_features
    self.use_pos_features = use_pos_features

    transformers = [
      (
        "counts",
        Pipeline([
          ("vectorizer", CountVectorizer(ngram_range=self.ngram_range)),
          ("normalizer", PerDocumentMaxNormalizer()),
        ]),
        "text",
      ),
    ]

    if self.use_positional_features:
      transformers.append((
          "positional",
          PositionalDistributionTransformer(ngram_range=self.ngram_range),
          "text",
      ))

    if self.use_pos_features:
      transformers.append((
        "pos_dense",
        StandardScaler(with_mean=False),
        DENSE_FEATURE_NAMES,
      ))

    self.model = Pipeline([
        ("features", ColumnTransformer(transformers)),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
            class_weight="balanced",
        )),
    ])

    
  def _to_dataframe(self, dataset):
    """Convert a list-of-dicts dataset into a DataFrame the
    ColumnTransformer can index by column name. Always provides 'text';
    conditionally adds the 17 dense feature columns when POS features are on."""
    rows = {"text": [" ".join(c["text"]) for c in dataset]}
    if self.use_pos_features:
      feats = np.vstack([
          _compute_dense_features(
              c.get("text", []),
              c.get("pos_tags", []),
              c.get("word_pos_pairs", []),
          )
          for c in dataset
      ])
      for i, name in enumerate(DENSE_FEATURE_NAMES):
        rows[name] = feats[:, i]
    return pd.DataFrame(rows)
  

  def fit(self, training_set):
    X_train = self._to_dataframe(training_set)
    Y_train = [conversation["manipulation_type"] for conversation in training_set]
    self.model.fit(X_train, Y_train)

  def predict(self, test_set, output_as_dict=False):
    X_test = self._to_dataframe(test_set)
    Y_test = [conversation["manipulation_type"] for conversation in test_set]
    Y_pred = self.model.predict(X_test)
    return classification_report(Y_test, Y_pred, zero_division=0, output_dict=output_as_dict)


  def feature_summary(self):
    """Returns dense POS coefficients when POS features are enabled.
    Walks the fitted ColumnTransformer to find where the dense block lives
    in the flat coefficient array, since other blocks may come before it."""
    if not self.use_pos_features:
      return {
          "n_word_features": 0,
          "n_pos_features": 0,
          "positive_class": None,
          "dense_pos_coefs": {},
      }

    feats = self.model.named_steps["features"]
    clf = self.model.named_steps["clf"]
    coefs = clf.coef_.ravel()

    offset = 0
    dense_coefs = None
    n_word = 0
    for name, transformer, _columns in feats.transformers_:
      if name == "remainder":
        continue
      block_width = self._block_width(name, transformer)
      if name == "counts":
        n_word = block_width
      if name == "pos_dense":
        dense_coefs = coefs[offset:offset + block_width]
        break
      offset += block_width

    classes = list(clf.classes_)
    return {
        "n_word_features": n_word,
        "n_pos_features": len(DENSE_FEATURE_NAMES),
        "positive_class": classes[1] if len(classes) == 2 else None,
        "dense_pos_coefs": (
            {name: float(c) for name, c in zip(DENSE_FEATURE_NAMES, dense_coefs)}
            if dense_coefs is not None else {}
        ),
    }

  @staticmethod
  def _block_width(name, transformer):
    """Width of a fitted transformer's output, used to find the dense
    block's offset in the flat coefficient vector."""
    if name == "counts":
      return len(transformer.named_steps["vectorizer"].vocabulary_)
    if name == "positional":
      return len(transformer.vocabulary_) * 3
    if name == "pos_dense":
      return len(DENSE_FEATURE_NAMES)
    return 0