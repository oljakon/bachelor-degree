from PyQt5 import uic, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
import sys
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from classification import get_train_data, svm_classification
from n_grams import generate_n_grams, generate_verb_unigrams_from_n_grams
from parse_text import read_text_from_file, split_text, lemmatize, get_pos_n_grams_string

morph = MorphAnalyzer()

n_table = {
    '2': 'datasets/pos_2_30.csv',
    '3': 'datasets/pos_3_30.csv',
    '4': 'datasets/pos_4_30.csv',
    '5': 'datasets/pos_5_30.csv',
    '6': 'datasets/pos_6_30.csv'
}

author_table = {
    'Chekhov': 'Чехов',
    'Tolstoy': 'Толстой',
    'Turgenev': 'Тургенев',
    'Gorky': 'Горький',
    'Dostoevsky': 'Достоевский'
}

global clf_svc, tfidf_vect
global filename


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi("mainwindow.ui", self)

    @pyqtSlot(name='on_fileSystem_clicked')
    def open_file_system(self):
        try:
            clf_svc, tfidf_vect
        except NameError:
            QMessageBox.warning(self, 'Ошибка', 'Сначала необходимо\nобучить классификатор!\n')
            return 0

        global filename
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        path_array = path.split('/')
        name = path_array[-1].split('.')
        self.name.setText(name[0])

    @pyqtSlot(name='on_findAuthor_clicked')
    def find_author(self):
        try:
            filename
        except NameError:
            QMessageBox.warning(self, 'Ошибка', 'Не выбран файл!\n')
            return 0

        path = filename[0]

        n = int(self.n.currentText())

        text = read_text_from_file(path)
        split = split_text(text)
        lemmatized_text = lemmatize(split, morph)
        n_gram_text = generate_n_grams(lemmatized_text, n)
        uni_pos_text = generate_verb_unigrams_from_n_grams(n_gram_text, morph)
        pos_n_grams_text = get_pos_n_grams_string(uni_pos_text)

        n_dataset = n_table[str(n)]

        pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data(n_dataset)

        try:
            path_array = path.split('/')
            author_path = path_array[-2]
            pos_text_train.append(pos_n_grams_text)
            pos_author_train.append(author_path)
            val = author_table[author_path]
        except KeyError:
            del pos_text_train[-1]
            del pos_author_train[-1]

        tfidf_vect = TfidfVectorizer()
        x_train = tfidf_vect.fit_transform(pos_text_train)
        x_test = tfidf_vect.transform(pos_text_test)

        clf_svc = SVC(C=2.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True,
                      tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                      decision_function_shape='ovo', break_ties=False, random_state=None)
        clf_svc.fit(x_train, pos_author_train)

        text_test = [pos_n_grams_text]
        text_test_tfidf = tfidf_vect.transform(text_test)
        y_pred = clf_svc.predict(text_test_tfidf)

        y_proba = clf_svc.predict_proba(text_test_tfidf)

        proba = '(' + str(round(100 * max(y_proba[0]))) + '%)'

        author_pred = author_table[y_pred[0]]

        # training_score = clf_svc.score(x_train, pos_author_train)
        # test_score = clf_svc.score(x_test, pos_author_test)
        # for doc, category in zip(pos_author_test, pred):
        #     print('%r => %s' % (doc, category))
        self.author.setText(author_pred + ' ' + proba)

    @pyqtSlot(name='on_classify_clicked')
    def classification(self):
        n = self.n.currentText()

        n_dataset = n_table[n]

        pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data(n_dataset)
        global clf_svc, tfidf_vect
        clf_svc, tfidf_vect, train_score, test_score, f_score = svm_classification(
            pos_text_train,
            pos_text_test,
            pos_author_train,
            pos_author_test
        )
        self.trainScore.setText(str(round(100 * train_score)) + '%')
        self.testScore.setText(str(round(100 * test_score)) + '%')
        self.fScore.setText(str(round(100 * f_score)) + '%')


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    main()
