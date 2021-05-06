from pymorphy2 import MorphAnalyzer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from n_grams import generate_n_grams, generate_pos_n_grams
from parse_text import split_text, lemmatize


def main():
    morph = MorphAnalyzer()

    data = pd.read_csv('author_dataset.csv', encoding='utf8')
    text = list(data['text'].values)
    author = list(data['author'].values)

    text_train, text_test, author_train, author_test = train_test_split(text, author, test_size=0.2, random_state=5)

    parsed_text = split_text('./text1.txt')
    lemmatized_text = lemmatize(parsed_text, morph)
    n_grams_array = generate_n_grams(lemmatized_text, 5)
    pos_n_grams_array = generate_pos_n_grams(n_grams_array, morph)

    #
    # parsed_text2 = split_text('./text2.txt', morph)
    # lemmatized_text2 = lemmatize(parsed_text2, morph)
    # n_grams_array2 = generate_n_grams(lemmatized_text2, 5)
    # pos_n_grams_array2 = generate_pos_n_grams(n_grams_array2, morph)
    #
    # corpus = [pos_n_grams_array, pos_n_grams_array2]
    # print(corpus)
    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(corpus)
    # print(X.shape)
    # vectorizer = CountVectorizer(ngram_range=(5, 5))
    # vectorizer.fit(pos_n_grams_array)
    # print(vectorizer.vocabulary_)
    # print(vectorizer.transform(pos_n_grams_array).toarray())


if __name__ == '__main__':
    main()