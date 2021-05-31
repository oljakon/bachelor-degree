from matplotlib import pyplot as plt

from classification import get_train_data, svm_classification

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

line3, = plt.plot(n, f)
plt.xlabel('N')
plt.ylabel('F-мера')
plt.grid()
plt.show()
