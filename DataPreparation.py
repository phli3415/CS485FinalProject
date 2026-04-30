import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import KFold
from sklearn.utils import resample


class DataPreparation:

  # This set of stop words can be remained to preserve important linguistic features that might be crucial for manipulation detection.
  # Please feel free to modify this list based on the specific requirements of your task.

  keep_words = {
      # Pronounce
      "i",
      "i'd",
      "i'll",
      "i'm",
      "i've",
      "me",
      "my",
      "mine",
      "myself",
      "you",
      "you'd",
      "you'll",
      "you're",
      "you've",
      "your",
      "yours",
      "yourself",
      "yourselves",
      "we",
      "we'd",
      "we'll",
      "we're",
      "we've",
      "our",
      "ours",
      "ourselves",
      "he",
      "he'd",
      "he'll",
      "he's",
      "him",
      "himself",
      "his",
      "she",
      "she'd",
      "she'll",
      "she's",
      "her",
      "hers",
      "herself",
      "it",
      "it'd",
      "it'll",
      "it's",
      "its",
      "itself",
      "they",
      "they'd",
      "they'll",
      "they're",
      "they've",
      "them",
      "themselves",
      "their",
      "theirs",
      # Negation
      "no",
      "nor",
      "not",
      "never",
      "none",
      "nothing",
      "nobody",
      "without",
      "ain",
      "ain't",
      "didn",
      "didn't",
      "doesn",
      "doesn't",
      "don",
      "don't",
      "hadn",
      "hadn't",
      "hasn",
      "hasn't",
      "haven",
      "haven't",
      "isn",
      "isn't",
      "wasn",
      "wasn't",
      "weren",
      "weren't",
      "won",
      "won't",
      "wouldn",
      "wouldn't",
      "couldn",
      "couldn't",
      "mightn",
      "mightn't",
      "mustn",
      "mustn't",
      "needn",
      "needn't",
      "shan",
      "shan't",
      # Modality / Emphasis
      "must",
      "should",
      "shouldn",
      "shouldn't",
      "should've",
      "would",
      "could",
      "might",
      "need",
      "can",
      "will",
      "very",
      "so",
      "too",
      "really",
      "just",
      "only",
      "simply",
      "more",
      "most",
      "all",
      "both",
      "even",
      "still",
      "already",
      # Contrast / Concession
      "but",
      "however",
      "although",
      "though",
      "yet",
      # Interrogatives
      "how",
      "what",
      "when",
      "where",
      "which",
      "who",
      "whom",
      "why",
      # Degree / Focus Particles
      "just",
      "only",
      "simply",
      "even",
      "still",
      "already",
      "so",
      "such",
      "too",
      # Demonstratives
      "this",
      "that",
      "these",
      "those",
      "it",
      "they",
      "them",
      "their",
      "theirs",
      # Specific Prepositions
      "for",
      "with",
      "without",
      "against",
      "like",
      "as",
      "than",
      "if",
      "then",
      "now",
      "own",
      "same",
      "such",
      "about",
      "because",
  }

  def __init__(self, file_path):
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    # nltk.download('stopwords')

    self.file_path = file_path
    self.data = []
    self.labels = []
    self.texts = []
    self.lemmatizer = nltk.stem.WordNetLemmatizer()
    self.stop_words = set(stopwords.words("english")) - self.keep_words
    self._pos_resources_ready = False

  def process_text(self, text):

    text = text.lower()
    tokens = text.split()

    words = [w for w in tokens if w not in self.stop_words]
    words = [re.sub(r"[^a-zA-Z\s]", "", w) for w in words]
    words = [self.lemmatizer.lemmatize(w) for w in words]
    words = [self.lemmatizer.lemmatize(w, pos="v") for w in words]

    return words
  

  def _ensure_pos_resources(self):
    if self._pos_resources_ready:
        return
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    self._pos_resources_ready = True


  def extract_word_pos_pairs(self, raw_text):
    self._ensure_pos_resources()

    if not raw_text or not isinstance(raw_text, str):
        return []

    tokens = nltk.word_tokenize(raw_text)
    return nltk.pos_tag(tokens)


  def extract_pos_tags(self, raw_text):
    return [tag for _, tag in self.extract_word_pos_pairs(raw_text)]
  

  def strip_speaker_labels(self, text):
    if not isinstance(text, str):
        return text
    return re.sub(r'\bPERSON\d+\s*:\s*', ' ', text, flags=re.IGNORECASE)
 


  def load_data(self, include_pos = False, strip_speakers=False, include_word_pos=False):
    self.texts = []

    if self.file_path.lower().endswith(".jsonl"):
      self.data = pd.read_json(self.file_path, lines=True)
      for _, conversation in self.data.iterrows():
        text = []

        pos_tags = []
        word_pos_pairs = []

        for msg in conversation["messages"]:
          raw = msg['text']
          if strip_speakers:
            raw = self.strip_speaker_labels(raw)
          text.extend(self.process_text(raw))
          if include_pos or include_word_pos:
            pairs = self.extract_word_pos_pairs(raw)
            if include_pos:
              pos_tags.extend(tag for _, tag in pairs)
            if include_word_pos:
              word_pos_pairs.extend(pairs)

        messages = {
            "manipulation_type": conversation["manipulation_type"],
            "is_manipulation": conversation["is_manipulation"],
            "text": text,
        }
        if include_pos:
          messages["pos_tags"] = pos_tags
        if include_word_pos:
          messages["word_pos_pairs"] = word_pos_pairs
        self.texts.append(messages)

    elif self.file_path.lower().endswith(".csv"):
      self.data = pd.read_csv(self.file_path)
      for _, row in self.data.iterrows():
        raw = str(row.get('dialogue', ''))
        if strip_speakers:
          raw = self.strip_speaker_labels(raw)

        text = self.process_text(raw)

        pos_tags = None
        word_pos_pairs = None
        if include_pos or include_word_pos:
          pairs = self.extract_word_pos_pairs(raw)
          if include_pos:
            pos_tags = [tag for _, tag in pairs]
          if include_word_pos:
            word_pos_pairs = pairs

        is_manipulation = self._csv_is_manipulation(row)
        manipulation_type = "manipulation" if is_manipulation else "neutral"

        messages = {
            "manipulation_type": manipulation_type,
            "is_manipulation": is_manipulation,
            "text": text,
        }
        if include_pos:
          messages["pos_tags"] = pos_tags
        if include_word_pos:
          messages["word_pos_pairs"] = word_pos_pairs
        self.texts.append(messages)

    else:
      raise ValueError(f"Unsupported file type: {self.file_path}")

    self.texts = self.deduplicate(self.texts)
    return self.texts

  def _csv_is_manipulation(self, row):
    for column in ["manipulative_1", "manipulative_2", "manipulative_3"]:
      value = row.get(column)
      if pd.notna(value):
        try:
          return int(value) == 1
        except (TypeError, ValueError):
          return str(value).strip() == "1"
    return False

  def deduplicate(self, texts):

    seen = set()
    unique_texts = []
    for record in texts:
      record_key = (
          record["manipulation_type"],
          record["is_manipulation"],
          " ".join(record["text"]),
      )
      if record_key in seen:
        continue
      seen.add(record_key)
      unique_texts.append(record)
    return unique_texts

  def cross_validation_split(self, k=5, shuffle=True, random_state=67):
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    dataset = {}

    for i, (train_index, test_index) in enumerate(kf.split(self.texts)):
      fold = {}
      fold["train"] = [self.texts[idx] for idx in train_index]
      fold["test"] = [self.texts[idx] for idx in test_index]
      dataset[f"fold_{i}"] = fold

    return dataset

  def shuffle_split(self, k=5, random_state=67):
    dataset = {}
    all_indices = np.arange(len(self.texts))

    for i in range(k):
      train_indices = resample(
          all_indices,
          replace=True,
          n_samples=10000,
          random_state=i + random_state,
      )
      test_indices = list(set(all_indices) - set(train_indices))
      fold = {}
      fold["train"] = [self.texts[idx] for idx in train_indices]
      fold["test"] = [self.texts[idx] for idx in test_indices]
      dataset[f"fold_{i}"] = fold

    return dataset
