from matplotlib import pyplot as plt

from classification import get_train_data, svm_classification


def experiment1():
    n = [2, 3, 4, 5, 6]
    train = []
    test = []
    f = []

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train.append(train_score2)
    test.append(test_score2)
    f.append(f2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train.append(train_score3)
    test.append(test_score3)
    f.append(f3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train.append(train_score4)
    test.append(test_score4)
    f.append(f4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train.append(train_score5)
    test.append(test_score5)
    f.append(f5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train.append(train_score6)
    test.append(test_score6)
    f.append(f6)

    print(train)
    print(test)
    print(f)

    line1, = plt.plot(n, train,  label='Обучающая выборка')
    line2, = plt.plot(n, test, label='Тестовая выборка')
    plt.xlabel('N')
    plt.ylabel('Точность')
    plt.legend(handles=[line1, line2])
    plt.grid()
    plt.show()

    plt.plot(n, f)
    plt.xlabel('N')
    plt.ylabel('F-мера')
    plt.grid()
    plt.show()


def experiment2():
    n = [2, 3, 4, 5, 6]
    train = []
    test = []
    f = []

    train_stop = []
    test_stop = []
    f_stop = []

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train.append(train_score2)
    test.append(test_score2)
    f.append(f2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train.append(train_score3)
    test.append(test_score3)
    f.append(f3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train.append(train_score4)
    test.append(test_score4)
    f.append(f4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train.append(train_score5)
    test.append(test_score5)
    f.append(f5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train.append(train_score6)
    test.append(test_score6)
    f.append(f6)

    ######################################

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_stop_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train_stop.append(train_score2)
    test_stop.append(test_score2)
    f_stop.append(f2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_stop_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train_stop.append(train_score3)
    test_stop.append(test_score3)
    f_stop.append(f3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_stop_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train_stop.append(train_score4)
    test_stop.append(test_score4)
    f_stop.append(f4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_stop_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train_stop.append(train_score5)
    test_stop.append(test_score5)
    f_stop.append(f5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_stop_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train_stop.append(train_score6)
    test_stop.append(test_score6)
    f_stop.append(f6)
    print(train)
    print(test)
    print(f)

    print(train_stop)
    print(test_stop)
    print(f_stop)

    line1, = plt.plot(n, train, label='Обучающая выборка')
    line2, = plt.plot(n, test, label='Тестовая выборка')
    line3, = plt.plot(n, train_stop, label='Обучающая выборка (без стоп-слов)')
    line4, = plt.plot(n, test_stop, label='Тестовая выборка (без стоп-слов)')
    plt.xlabel('N')
    plt.ylabel('Точность')
    plt.legend(handles=[line1, line2, line3, line4])
    plt.grid()
    plt.show()

    line5, = plt.plot(n, f, label='Выборка со стоп-словами')
    line6, = plt.plot(n, f_stop, label='Выборка без стоп-слов')
    plt.plot(n, f)
    plt.xlabel('N')
    plt.ylabel('F-мера')
    plt.legend(handles=[line5, line6])
    plt.grid()
    plt.show()


def experiment3():
    n = [2, 3, 4, 5, 6]
    train = []
    test = []
    f = []

    train_verb = []
    test_verb = []
    f_verb = []

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train.append(train_score2)
    test.append(test_score2)
    f.append(f2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train.append(train_score3)
    test.append(test_score3)
    f.append(f3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train.append(train_score4)
    test.append(test_score4)
    f.append(f4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train.append(train_score5)
    test.append(test_score5)
    f.append(f5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train.append(train_score6)
    test.append(test_score6)
    f.append(f6)

    ######################################

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_verb_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train_verb.append(train_score2)
    test_verb.append(test_score2)
    f_verb.append(f2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_verb_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train_verb.append(train_score3)
    test_verb.append(test_score3)
    f_verb.append(f3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_verb_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train_verb.append(train_score4)
    test_verb.append(test_score4)
    f_verb.append(f4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_verb_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train_verb.append(train_score5)
    test_verb.append(test_score5)
    f_verb.append(f5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_verb_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
    train_verb.append(train_score6)
    test_verb.append(test_score6)
    f_verb.append(f6)
    print(train)
    print(test)
    print(f)

    print(train_verb)
    print(test_verb)
    print(f_verb)

    line1, = plt.plot(n, train, label='Обучающая выборка')
    line2, = plt.plot(n, test, label='Тестовая выборка')
    line3, = plt.plot(n, train_verb, label='Обучающая выборка (без инфинитивов)')
    line4, = plt.plot(n, test_verb, label='Тестовая выборка (без инфинитивов)')
    plt.xlabel('N')
    plt.ylabel('Точность')
    plt.legend(handles=[line1, line2, line3, line4])
    plt.grid()
    plt.show()

    line5, = plt.plot(n, f, label='Выборка с инфинитивами')
    line6, = plt.plot(n, f_verb, label='Выборка без инфинитивов')
    plt.plot(n, f)
    plt.xlabel('N')
    plt.ylabel('F-мера')
    plt.legend(handles=[line5, line6])
    plt.grid()
    plt.show()


def experiment41():
    n = [2, 3, 4, 5, 6]
    train_lin = []

    train_lin = []
    test_lin = []
    f_lin = []

    train_poly = []
    test_poly = []
    f_poly = []

    train_rbf = []
    test_rbf = []
    f_rbf = []

    train_sig = []
    test_sig = []
    f_sig = []

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'linear')
    train_lin.append(train_score2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'linear')
    train_lin.append(train_score3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'linear')
    train_lin.append(train_score4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'linear')
    train_lin.append(train_score5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'linear')
    train_lin.append(train_score6)

    ###########################################

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'poly')
    train_poly.append(train_score2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'poly')
    train_poly.append(train_score3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'poly')
    train_poly.append(train_score4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'poly')
    train_poly.append(train_score5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'poly')
    train_poly.append(train_score6)

    ###########################################

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'rbf')
    train_rbf.append(train_score2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'rbf')
    train_rbf.append(train_score3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'rbf')
    train_rbf.append(train_score4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'rbf')
    train_rbf.append(train_score5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'rbf')
    train_rbf.append(train_score6)

    ###########################################

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'sigmoid')
    train_sig.append(train_score2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'sigmoid')
    train_sig.append(train_score3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'sigmoid')
    train_sig.append(train_score4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'sigmoid')
    train_sig.append(train_score5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'sigmoid')
    train_sig.append(train_score6)

    line1, = plt.plot(n, train_lin, label='Линейное')
    line2, = plt.plot(n, train_poly, label='Полиномиальное')
    line3, = plt.plot(n, train_rbf, label='Гауссово RBF')
    line4, = plt.plot(n, train_sig, label='Сигмоидальное')
    plt.xlabel('N')
    plt.ylabel('Точность на обучающей выборке')
    plt.legend(handles=[line1, line2, line3, line4])
    plt.grid()
    plt.show()


def experiment42():
    n = [2, 3, 4, 5, 6]
    train_lin = []

    train_lin = []
    test_lin = []
    f_lin = []

    train_poly = []
    test_poly = []
    f_poly = []

    train_rbf = []
    test_rbf = []
    f_rbf = []

    train_sig = []
    test_sig = []
    f_sig = []

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'linear')
    test_lin.append(test_score2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'linear')
    test_lin.append(test_score3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'linear')
    test_lin.append(test_score4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'linear')
    test_lin.append(test_score5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'linear')
    test_lin.append(test_score6)

    ###########################################

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'poly')
    test_poly.append(test_score2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'poly')
    test_poly.append(test_score3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'poly')
    test_poly.append(test_score4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'poly')
    test_poly.append(test_score5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'poly')
    test_poly.append(test_score6)

    ###########################################

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'rbf')
    test_rbf.append(test_score2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'rbf')
    test_rbf.append(test_score3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'rbf')
    test_rbf.append(test_score4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'rbf')
    test_rbf.append(test_score5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'rbf')
    test_rbf.append(test_score6)

    ###########################################

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'sigmoid')
    test_sig.append(test_score2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'sigmoid')
    test_sig.append(test_score3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'sigmoid')
    test_sig.append(test_score4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'sigmoid')
    test_sig.append(test_score5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'sigmoid')
    test_sig.append(test_score6)

    line1, = plt.plot(n, test_lin, label='Линейное')
    line2, = plt.plot(n, test_poly, label='Полиномиальное')
    line3, = plt.plot(n, test_rbf, label='Гауссово RBF')
    line4, = plt.plot(n, test_sig, label='Сигмоидальное')
    plt.xlabel('N')
    plt.ylabel('Точность на тестовой выборке')
    plt.legend(handles=[line1, line2, line3, line4])
    plt.grid()
    plt.show()


def experiment43():
    n = [2, 3, 4, 5, 6]
    train_lin = []

    train_lin = []
    f_lin = []

    train_poly = []
    f_poly = []

    train_rbf = []
    f_rbf = []

    train_sig = []
    f_sig = []

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'linear')
    f_lin.append(f2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'linear')
    f_lin.append(f3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'linear')
    f_lin.append(f4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'linear')
    f_lin.append(f5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'linear')
    f_lin.append(f6)

    ###########################################

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'poly')
    f_poly.append(f2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'poly')
    f_poly.append(f3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'poly')
    f_poly.append(f4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'poly')
    f_poly.append(f5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test, 'poly')
    f_poly.append(f6)

    ###########################################

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'rbf')
    f_rbf.append(f2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'rbf')
    f_rbf.append(f3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'rbf')
    f_rbf.append(f4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'rbf')
    f_rbf.append(f5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'rbf')
    f_rbf.append(f6)

    ###########################################

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_30.csv')
    train_score2, test_score2, f2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'sigmoid')
    f_sig.append(f2)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
    train_score3, test_score3, f3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'sigmoid')
    f_sig.append(f3)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_30.csv')
    train_score4, test_score4, f4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'sigmoid')
    f_sig.append(f4)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_30.csv')
    train_score5, test_score5, f5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'sigmoid')
    f_sig.append(f5)

    pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_30.csv')
    train_score6, test_score6, f6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test,
                                                       'sigmoid')
    f_sig.append(f6)

    line1, = plt.plot(n, f_lin, label='Линейное')
    line2, = plt.plot(n, f_poly, label='Полиномиальное')
    line3, = plt.plot(n, f_rbf, label='Гауссово RBF')
    line4, = plt.plot(n, f_sig, label='Сигмоидальное')
    plt.xlabel('N')
    plt.ylabel('F-мера')
    plt.legend(handles=[line1, line2, line3, line4])
    plt.grid()
    plt.show()


def main():
    experiment43()


if __name__ == '__main__':
    main()
