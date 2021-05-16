import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def main():
    pos_data = pd.read_csv('author_pos.csv', encoding='utf8')
    pos_text = list(pos_data['pos'].values)
    pos_author = list(pos_data['author'].values)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = train_test_split(
        pos_text, pos_author, test_size=0.2, random_state=5
    )

    # Convert a collection of text documents to a matrix of token counts
    vect = CountVectorizer()
    x_train = vect.fit_transform(pos_text_train)

    #print(vect.vocabulary_)
    #print(vect.get_feature_names())
    #print(x_train.toarray())

    tfidf_vect = TfidfVectorizer()
    x_train = tfidf_vect.fit_transform(pos_text_train)

    #print(x_train.toarray())

    # tfidf_transformer = TfidfTransformer()
    # x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    #
    # model = MultinomialNB()
    # model_train = model.fit(x_train_tfidf, pos_author_train)
    # training_score = model.score(x_train_tfidf, pos_author_train)
    #
    # x_test_counts = count_vect.transform(pos_text_test)
    # x_test_tfidf = tfidf_transformer.transform(x_test_counts)
    #
    # predicted = model.predict(x_test_tfidf)
    # test_score = model.score(x_test_tfidf, pos_author_test)
    #
    # print(f'Training score: {training_score}, \nTest score: {test_score}')

    # for doc, category in zip(pos_text_test, predicted):
    #     print('%r => %s' % (doc, category))


if __name__ == '__main__':
    main()