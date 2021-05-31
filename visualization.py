from matplotlib import pyplot as plt

from classification import get_train_data, svm_classification

n = [2, 3, 4, 5, 6]
train = []
test = []

pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_2_30.csv')
train_score2, test_score2 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
train.append(train_score2)
test.append(test_score2)

pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_3_30.csv')
train_score3, test_score3 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
train.append(train_score3)
test.append(test_score3)

pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_4_30.csv')
train_score4, test_score4 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
train.append(train_score4)
test.append(test_score4)

pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_5_30.csv')
train_score5, test_score5 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
train.append(train_score5)
test.append(test_score5)

pos_text_train, pos_text_test, pos_author_train, pos_author_test = get_train_data('datasets/pos_6_30.csv')
train_score6, test_score6 = svm_classification(pos_text_train, pos_text_test, pos_author_train, pos_author_test)
train.append(train_score6)
test.append(test_score6)

print(train)
print(test)

line1, = plt.plot(n, train,  label='Обучающая выборка')
line2, = plt.plot(n, test, label='Тестовая выборка')
plt.xlabel('N')
plt.ylabel('Точность')
plt.legend(handles=[line1, line2])
plt.grid()
plt.show()
