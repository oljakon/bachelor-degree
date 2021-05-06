from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from dataset import author_dataset
from n_grams import generate_n_grams, generate_pos_n_grams
from parse_text import split_text, lemmatize


def main():
    morph = MorphAnalyzer()
    parsed_text = split_text('./text1.txt', morph)
    lemmatized_text = lemmatize(parsed_text, morph)
    n_grams_array = generate_n_grams(lemmatized_text, 5)
    pos_n_grams_array = generate_pos_n_grams(n_grams_array, morph)

    parsed_text2 = split_text('./text2.txt', morph)
    lemmatized_text2 = lemmatize(parsed_text2, morph)
    n_grams_array2 = generate_n_grams(lemmatized_text2, 5)
    pos_n_grams_array2 = generate_pos_n_grams(n_grams_array2, morph)

    corpus = [pos_n_grams_array, pos_n_grams_array2]
    print(corpus)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    print(X.shape)
    # vectorizer = CountVectorizer(ngram_range=(5, 5))
    # vectorizer.fit(pos_n_grams_array)
    # print(vectorizer.vocabulary_)
    # print(vectorizer.transform(pos_n_grams_array).toarray())


if __name__ == '__main__':
    main()