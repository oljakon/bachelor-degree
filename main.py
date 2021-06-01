from PyQt5 import uic, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import sys
from pymorphy2 import MorphAnalyzer
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


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi("mainwindow.ui", self)

    @pyqtSlot(name='on_fileSystem_clicked')
    def open_file_system(self):
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        path_array = path.split('/')
        self.name.setText(path_array[-1])

        clf_svc, tfidf_vect = self.classification()

        text = read_text_from_file(path)
        split = split_text(text)
        lemmatized_text = lemmatize(split, morph)
        n_gram_text = generate_n_grams(lemmatized_text, 3)
        uni_pos_text = generate_verb_unigrams_from_n_grams(n_gram_text, morph)
        pos_n_grams_text = get_pos_n_grams_string(uni_pos_text)

        text_test = [pos_n_grams_text]
        author = tfidf_vect.transform(text_test)
        y_pred = clf_svc.predict(author)

        author_pred = author_table[y_pred[0]]

        self.author.setText(author_pred)



    @pyqtSlot(name='on_classify_clicked')
    def classification(self):
        n = self.n.currentText()

        n_dataset = n_table[n]

        pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data(n_dataset)
        clf_svc, tfidf, train_score, test_score, f_score = svm_classification(
            pos_text_train,
            pos_text_test,
            pos_author_train,
            pos_author_test
        )
        self.trainScore.setText(str(round(100 * train_score)) + '%')
        self.testScore.setText(str(round(100 * test_score)) + '%')
        self.fScore.setText(str(round(100 * f_score)) + '%')

        return clf_svc, tfidf


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    main()
