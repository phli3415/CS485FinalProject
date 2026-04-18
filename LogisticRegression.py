import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

class LogisticRegressionModel:
    def __init__ (self, random_state = 67, ngram_range=(1,1)):
        self.random_state = random_state
        self.ngram_range = ngram_range
        self.model = Pipeline([
            ('vectorizer', CountVectorizer(ngram_range=self.ngram_range)),
            ('clf', LogisticRegression(solver='lbfgs', max_iter=1000, random_state=self.random_state))        
        ])


    def fit(self, training_set):
        X_train = [' '.join(conversation['text']) for conversation in training_set]
        Y_train = [conversation['manipulation_type'] for conversation in training_set]
        self.model.fit(X_train, Y_train)
    
    def predict(self, test_set):
        X_test = [' '.join(conversation['text']) for conversation in test_set]
        Y_test = [conversation['manipulation_type'] for conversation in test_set]
        Y_pred = self.model.predict(X_test)
        return classification_report(Y_test, Y_pred, zero_division=0)


