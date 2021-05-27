import pandas as pd
from pymorphy2 import MorphAnalyzer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from classification import get_train_data, svm_classification
from n_grams import generate_n_grams, generate_pos_unigrams_from_n_grams
from parse_text import read_text_from_file, split_text, lemmatize, get_pos_n_grams_string


def main():
    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('author_pos_3_30.csv')
    svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)


if __name__ == '__main__':
    main()
    # morph = MorphAnalyzer()
    #
    # text = read_text_from_file('text.txt')
    # split = split_text(text)
    # lemmatized_text = lemmatize(split, morph)
    # n_gram_text = generate_n_grams(lemmatized_text, 3)
    # uni_pos_text = generate_pos_unigrams_from_n_grams(n_gram_text, morph)
    # print(uni_pos_text)
    # pos_n_grams_text = get_pos_n_grams_string(uni_pos_text)
