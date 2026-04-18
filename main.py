from DataPreparation import DataPreparation

from nltk.corpus import stopwords

from LogisticRegression import LogisticRegressionModel



if __name__ == "__main__":
    # print(stopwords.words('english'))

    data_prep = DataPreparation("data/mentalmanip_detailed.csv")
    texts = data_prep.load_data()

    dataset = data_prep.cross_validation_split(k=5)




    logistic_regression_unigram = LogisticRegressionModel(ngram_range=(1,1))
    logistic_regression_bigram = LogisticRegressionModel(ngram_range=(1, 2))

    # cross validation
    for fold, data in dataset.items():
        print(f"Fold: {fold} Length of Train: {len(data['train'])} Length of Test: {len(data['test'])}")
        logistic_regression_unigram.fit(data['train'])
        logistic_regression_bigram.fit(data['train'])

        print(f"Fold: {fold} - Unigram Logistic Regression")
        print(logistic_regression_unigram.predict(data['test']))
        print(f"Fold: {fold} - Bigram Logistic Regression")
        print(logistic_regression_bigram.predict(data['test']))


        


