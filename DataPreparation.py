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
        'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'me', 'my', 'mine', 'myself',
        'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves',
        'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'our', 'ours', 'ourselves',
        'he', 'he\'d', 'he\'ll', 'he\'s', 'him', 'himself', 'his',
        'she', 'she\'d', 'she\'ll', 'she\'s', 'her', 'hers', 'herself',
        'it', 'it\'d', 'it\'ll', 'it\'s', 'its', 'itself',
        'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'them', 'themselves', 'their', 'theirs',
   
        # Negation
        'no', 'nor', 'not', 'never', 'none', 'nothing', 'nobody', 'without', 'ain', 'ain\'t',
        'didn', 'didn\'t', 'doesn', 'doesn\'t', 'don', 'don\'t', 'hadn', 'hadn\'t', 'hasn', 'hasn\'t',
        'haven', 'haven\'t', 'isn', 'isn\'t', 'wasn', 'wasn\'t', 'weren', 'weren\'t', 'won', 'won\'t',
        'wouldn', 'wouldn\'t', 'couldn', 'couldn\'t', 'mightn', 'mightn\'t', 'mustn', 'mustn\'t',
        'needn', 'needn\'t', 'shan', 'shan\'t',

        # Modality / Emphasis
        'must', 'should', 'shouldn', 'shouldn\'t', 'should\'ve', 'would', 'could', 'might', 'need', 'can', 'will',
        'very', 'so', 'too', 'really', 'just', 'only', 'simply', 'more', 'most', 'all', 'both', 'even', 'still', 'already',

        # Contrast / Concession
        'but', 'however', 'although', 'though', 'yet',

        # Interrogatives
        'how', 'what', 'when', 'where', 'which', 'who', 'whom', 'why',

        # Degree / Focus Particles
        'just', 'only', 'simply', 'even', 'still', 'already', 'so', 'such', 'too',

        # Demonstratives
        'this', 'that', 'these', 'those', 'it', 'they', 'them', 'their', 'theirs',
        
        # Specific Prepositions
        'for', 'with', 'without', 'against', 'like', 'as', 'than', 'if', 'then', 'now', 'own', 'same', 'such', 'about', 'because'
    }

    def __init__(self, file_path):
        # nltk.download('wordnet')
        # nltk.download('omw-1.4')
        self.file_path = file_path
        self.data = None
        self.labels = None
        self.texts = None
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))-self.keep_words

    def process_text (self,text):

        text = text.lower()
        tokens = text.split()

        words = [w for w in tokens if w not in self.stop_words]
        words = [re.sub(r'[^a-zA-Z\s]', '', w) for w in words]
        words = [self.lemmatizer.lemmatize(w) for w in words]
        words = [self.lemmatizer.lemmatize(w, pos='v') for w in words]

        return words


    def load_data(self):
        self.data = pd.read_json(self.file_path, lines=True)
        self.texts = []
        for _,conversation in self.data.iterrows():
            text = []
            for msg in conversation['messages']:
                text.extend(self.process_text(msg['text']))

            messages = {
                "manipulation_type": conversation['manipulation_type'],
                "is_manipulation": conversation['is_manipulation'],
                "text": text
            }

            self.texts.append(messages)

        self.texts = self.deduplicate(self.texts)
        return self.texts

    def deduplicate(self, texts):

        seen = set()
        unique_texts = []
        for record in texts:
            record_key = (
                record['manipulation_type'],
                record['is_manipulation'],
                ' '.join(record['text'])
            )
            if record_key in seen:
                continue
            seen.add(record_key)
            unique_texts.append(record)
        return unique_texts
    
        
    def cross_validation_split(self, k = 5, shuffle = True, random_state = 67):
        kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        dataset = {}

        for i, (train_index, test_index) in enumerate(kf.split(self.texts)):
            fold = {}
            fold['train'] = [self.texts[idx] for idx in train_index]
            fold['test'] = [self.texts[idx] for idx in test_index]
            dataset[f'fold_{i}'] = fold
        
        return dataset
        

    def shuffle_split(self, k = 5, random_state = 67):
        dataset = {}
        all_indices = np.arange(len(self.texts))

        for i in range (k):
            train_indices = resample(all_indices, replace=True, n_samples=10000, random_state=i+random_state)
            test_indices = list(set(all_indices) - set(train_indices))
            fold = {}
            fold['train'] = [self.texts[idx] for idx in train_indices]
            fold['test'] = [self.texts[idx] for idx in test_indices]
            dataset[f'fold_{i}'] = fold

        return dataset
