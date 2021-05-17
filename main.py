import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


def main():
    pos_data = pd.read_csv('author_pos_5.csv', encoding='utf8')
    pos_text = list(pos_data['pos'].values)
    pos_author = list(pos_data['author'].values)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = train_test_split(
        pos_text, pos_author, test_size=0.2, random_state=5
    )

    # Convert a collection of text documents to a matrix of token counts
    vect = CountVectorizer()
    x_train_count = vect.fit_transform(pos_text_train)
    x_test_count = vect.transform(pos_text_test)

    # print(vect.vocabulary_)
    # print(vect.get_feature_names())
    # print(x_train_count.toarray())

    tfidf_vect = TfidfVectorizer()
    x_train = tfidf_vect.fit_transform(pos_text_train)
    x_test = tfidf_vect.transform(pos_text_test)

    # print(x_train.toarray())

    # clf_mnb = MultinomialNB(alpha=0.0001, fit_prior=False)
    # clf_mnb.fit(x_train, pos_author_train)
    # y_pred = clf_mnb.predict(x_test)
    # mnb_score = accuracy_score(pos_author_test, y_pred)
    #
    # training_score = clf_mnb.score(x_train, pos_author_train)
    # test_score = clf_mnb.score(x_test, pos_author_test)
    #
    # print(f'TfIdfVectorizer:\nTraining score: {training_score}, \nTest score: {test_score}\n')
    #
    # clf_mnb_count = MultinomialNB()
    # clf_mnb_count.fit(x_train_count, pos_author_train)
    # y_pred_count = clf_mnb.predict(x_test_count)
    # mnb_score_count = accuracy_score(pos_author_test, y_pred_count)
    #
    # training_score_count = clf_mnb_count.score(x_train_count, pos_author_train)
    # test_score_count = clf_mnb_count.score(x_test_count, pos_author_test)
    #
    # print(f'CountVectorizer:\nTraining score: {training_score_count}, \nTest score: {test_score_count}')

    clf_neigh = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                     metric='minkowski', metric_params=None, n_jobs=None)
    clf_neigh.fit(x_train, pos_author_train)
    y_pred = clf_neigh.predict(x_test)
    mnb_score = accuracy_score(pos_author_test, y_pred)

    training_score = clf_neigh.score(x_train, pos_author_train)
    test_score = clf_neigh.score(x_test, pos_author_test)

    print(f'TfIdfVectorizer:\nTraining score: {training_score}, \nTest score: {test_score}\n')

    for doc, category in zip(pos_author_test, y_pred):
        print('%r => %s' % (doc, category))


if __name__ == '__main__':
    main()