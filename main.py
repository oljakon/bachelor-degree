from PyQt5 import uic, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QLineEdit, QTableWidgetItem, QFileDialog
import sys
import nltk
import pandas as pd
from pymorphy2 import MorphAnalyzer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from classification import get_train_data, svm_classification
from n_grams import generate_n_grams, generate_pos_unigrams_from_n_grams, generate_verb_unigrams_from_n_grams
from parse_text import read_text_from_file, split_text, lemmatize, get_pos_n_grams_string, lemmatize_remove_stopwords


# def main():
#     pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
#     train_score, test_score, f1 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
#
#
# if __name__ == '__main__':
#     main()
#
#     morph = MorphAnalyzer()
#
#     text = read_text_from_file('text.txt')
#     split = split_text(text)
#     lemmatized_text = lemmatize(split, morph)
#     lemmatized_text_stop = lemmatize_remove_stopwords(split, morph)
#     n_gram_text = generate_n_grams(lemmatized_text, 3)
#     uni_pos_text = generate_verb_unigrams_from_n_grams(n_gram_text, morph)
#     print(uni_pos_text)
#     pos_n_grams_text = get_pos_n_grams_string(uni_pos_text)

n_table = {
    '2': 'datasets/pos_2_30.csv',
    '3': 'datasets/pos_3_30.csv',
    '4': 'datasets/pos_4_30.csv',
    '5': 'datasets/pos_5_30.csv',
    '6': 'datasets/pos_6_30.csv'
}


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi("mainwindow.ui", self)

    @pyqtSlot(name='on_fileSystem_clicked')
    def open_file_system(self):
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        print(path)

    @pyqtSlot(name='on_classify_clicked')
    def classification(self):
        n = self.n.currentText()

        n_dataset = n_table[n]

        pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data(n_dataset)
        train_score, test_score, f1 = svm_classification(
            pos_text_train,
            pos_text_test,
            pos_author_train,
            pos_author_test
        )

        print(train_score, test_score, f1)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    main()
