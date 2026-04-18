from DataPreparation import DataPreparation
import numpy as np

prep = DataPreparation('data/manipulational_conversation.jsonl')
prep.load_data()
dataset = prep.cross_validation_split(k=5)
print('len', len(prep.texts))
for fold, data in dataset.items():
    print(fold, len(data['train']), len(data['test']))
    train_texts = [' '.join(item['text']) for item in data['train']]
    test_texts = [' '.join(item['text']) for item in data['test']]
    duplicate_texts = [text for text in test_texts if text in set(train_texts)]
    print(' duplicate count', len(duplicate_texts))
    # get first 5 unique duplicate texts
    seen = set()
    first5 = []
    for text in duplicate_texts:
        if text not in seen:
            seen.add(text)
            first5.append(text)
        if len(first5) == 5:
            break
    for idx, text in enumerate(first5, 1):
        print(f'  dup_{idx}: {text}')
