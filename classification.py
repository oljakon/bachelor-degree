import pandas as pd
from matplotlib import pyplot as plt
from pymorphy2 import MorphAnalyzer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def get_train_data(csv_filename: str):
    pos_data = pd.read_csv(csv_filename, encoding='utf8')
    pos_text = list(pos_data['pos'].values)
    # su = 0
    # for i in pos_text:
    #     su += len(i)
    # print(su)
    pos_author = list(pos_data['author'].values)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = train_test_split(
        pos_text, pos_author, test_size=0.2, random_state=5
    )

    return pos_text_train, pos_text_test, pos_author_train, pos_author_test


def svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, kernel='rbf'):
    # sublinear_tf=True
    tfidf_vect = TfidfVectorizer()
    x_train = tfidf_vect.fit_transform(pos_text_train)
    x_test = tfidf_vect.transform(pos_text_test)

    clf_svc = SVC(C=2.0, kernel=kernel, degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True,
                  tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                  decision_function_shape='ovo', break_ties=False, random_state=None)
    clf_svc.fit(x_train, pos_author_train)
    # print(clf_svc.predict_proba(x_test))
    y_pred = clf_svc.predict(x_test)
    mnb_score = accuracy_score(pos_author_test, y_pred)

    training_score = clf_svc.score(x_train, pos_author_train)
    test_score = clf_svc.score(x_test, pos_author_test)

    f1 = f1_score(pos_author_test, y_pred, average='weighted')

    # print(f'TfIdfVectorizer:\nTraining score: {training_score}, \nTest score: {test_score}\n')
    # print(f'F1_score: {f1}\n')

    # for doc, category in zip(pos_author_test, y_pred):
    #     print('%r => %s' % (doc, category))

    return training_score, test_score, f1

    # Convert a collection of text documents to a matrix of token counts
    # vect = CountVectorizer()
    # x_train_count = vect.fit_transform(pos_text_train)
    # x_test_count = vect.transform(pos_text_test)

    # print(vect.vocabulary_)
    # print(vect.get_feature_names())
    # print(x_train_count.toarray())

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

    # clf_svc = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2,
    #                                  metric='minkowski', metric_params=None, n_jobs=None)
    # clf_svc.fit(x_train, pos_author_train)
    # y_pred = clf_svc.predict(x_test)
    # mnb_score = accuracy_score(pos_author_test, y_pred)
    #
    # training_score = clf_svc.score(x_train, pos_author_train)
    # test_score = clf_svc.score(x_test, pos_author_test)
    #
    # print(f'TfIdfVectorizer:\nTraining score: {training_score}, \nTest score: {test_score}\n')



    # clf_rf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None, min_samples_split=2,
    #                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
    #                                 max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
    #                                 bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
    #                                 warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
    #
    # clf_rf.fit(x_train, pos_author_train)
    # y_pred = clf_rf.predict(x_test)
    # mnb_score = accuracy_score(pos_author_test, y_pred)
    #
    # training_score = clf_rf.score(x_train, pos_author_train)
    # test_score = clf_rf.score(x_test, pos_author_test)
    #
    # print(f'TfIdfVectorizer:\nTraining score: {training_score}, \nTest score: {test_score}\n')

    # clf_lr = LogisticRegression(random_state=0)
    # clf_lr.fit(x_train, pos_author_train)
    # y_pred = clf_lr.predict(x_test)
    # mnb_score = accuracy_score(pos_author_test, y_pred)
    #
    # training_score = clf_lr.score(x_train, pos_author_train)
    # test_score = clf_lr.score(x_test, pos_author_test)
    #
    # print(f'TfIdfVectorizer:\nTraining score: {training_score}, \nTest score: {test_score}\n')