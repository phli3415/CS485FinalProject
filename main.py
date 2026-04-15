from DataPreparation import DataPreparation

from nltk.corpus import stopwords



if __name__ == "__main__":
    # print(stopwords.words('english'))

    data_prep = DataPreparation("data/manipulational_conversation.jsonl")
    # print (data_prep.stop_words)
    texts = data_prep.load_data()
    # print(texts[:30])

    dataset = data_prep.shuffle_split(k=5)
    for fold, data in dataset.items():
        # print(f"{fold}: Train set: {data['train'][:10]}")

        # print("-----------------------------")
        # print("test set: ", data['test'][:10])

        print(f"{fold}: Train size: {len(data['train'])}, Test size: {len(data['test'])}")