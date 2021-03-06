import glob
import random
import numpy as np
import pandas as pd
from pymorphy2 import MorphAnalyzer

from n_grams import generate_n_grams, generate_pos_unigrams_from_n_grams, generate_verb_unigrams_from_n_grams
from parse_text import read_text_from_file, split_text, lemmatize, get_pos_n_grams_string, lemmatize_remove_stopwords


chekhov = []
for path in glob.glob('./prose/Chekhov/*.txt'):
    chekhov.append(read_text_from_file(path))

dostoevsky = []
for path in glob.glob('./prose/Dostoevsky/*.txt'):
    dostoevsky.append(read_text_from_file(path))

tolstoy = []
for path in glob.glob('./prose/Tolstoy/*.txt'):
    tolstoy.append(read_text_from_file(path))

gorky = []
for path in glob.glob('./prose/Gorky/*.txt'):
    gorky.append(read_text_from_file(path))

turgenev = []
for path in glob.glob('./prose/Turgenev/*.txt'):
    turgenev.append(read_text_from_file(path))

max_len = 30

chekhov = chekhov[:max_len]
dostoevsky = dostoevsky[:max_len]
tolstoy = tolstoy[:max_len]
gorky = gorky[:max_len]
turgenev = turgenev[:max_len]

names = [chekhov, dostoevsky, tolstoy, gorky, turgenev]

morph = MorphAnalyzer()

# chekhov_n = []
# n_dict = {}
# for text in chekhov:
#     parsed_text = split_text(text)
#     lemmatized_text = lemmatize(parsed_text, morph)
#     n_gram_text = generate_n_grams(lemmatized_text, 3)
#     uni_pos_text = generate_verb_unigrams_from_n_grams(n_gram_text, morph)
#     pos_n_grams_text = get_pos_n_grams_string(uni_pos_text)
#     chekhov_n.append(pos_n_grams_text)

# n_string = chekhov_n[0].split(' ')
# for n_gram in n_string:
#     if n_gram in n_dict:
#         n_dict[n_gram] += 1
#     else:
#         n_dict.update({n_gram: 1})
#
# sorted_tuples = sorted(n_dict.items(), key=lambda item: item[1])
# print(sorted_tuples)  # [(1, 1), (3, 4), (2, 9)]
#


combined = []
for name in names:
    name = np.random.choice(name, max_len, replace=False)
    combined += list(name)

labels = ['Chekhov'] * max_len + ['Dostoevsky'] * max_len + ['Tolstoy'] * max_len + ['Gorky'] * max_len \
         + ['Turgenev'] * max_len

random.seed(3)

zipped = list(zip(combined, labels))
random.shuffle(zipped)
combined, labels = zip(*zipped)

author_text = pd.DataFrame()
author_text['text'] = combined
author_text['author'] = labels

author_text.to_csv('datasets/author_text_30.csv', index=False)

morph = MorphAnalyzer()


pos_2_grams_dataset = []
for text in combined:
    parsed_text = split_text(text)
    lemmatized_text = lemmatize(parsed_text, morph)
    n_gram_text = generate_n_grams(lemmatized_text, 2)
    uni_pos_text = generate_pos_unigrams_from_n_grams(n_gram_text, morph)
    pos_n_grams_text = get_pos_n_grams_string(uni_pos_text)
    pos_2_grams_dataset.append(pos_n_grams_text)
author_pos_3 = pd.DataFrame()
author_pos_3['pos'] = pos_2_grams_dataset
author_pos_3['author'] = labels
author_pos_3.to_csv('datasets/pos_2_30.csv', index=False)

pos_3_grams_dataset = []
for text in combined:
    parsed_text = split_text(text)
    lemmatized_text = lemmatize(parsed_text, morph)
    n_gram_text = generate_n_grams(lemmatized_text, 3)
    uni_pos_text = generate_pos_unigrams_from_n_grams(n_gram_text, morph)
    pos_n_grams_text = get_pos_n_grams_string(uni_pos_text)
    pos_3_grams_dataset.append(pos_n_grams_text)
author_pos_3 = pd.DataFrame()
author_pos_3['pos'] = pos_3_grams_dataset
author_pos_3['author'] = labels
author_pos_3.to_csv('datasets/pos_3_30.csv', index=False)

pos_4_grams_dataset = []
for text in combined:
    parsed_text = split_text(text)
    lemmatized_text = lemmatize(parsed_text, morph)
    n_gram_text = generate_n_grams(lemmatized_text, 4)
    uni_pos_text = generate_pos_unigrams_from_n_grams(n_gram_text, morph)
    pos_n_grams_text = get_pos_n_grams_string(uni_pos_text)
    pos_4_grams_dataset.append(pos_n_grams_text)
author_pos_4 = pd.DataFrame()
author_pos_4['pos'] = pos_4_grams_dataset
author_pos_4['author'] = labels
author_pos_4.to_csv('datasets/pos_4_30.csv', index=False)

pos_5_grams_dataset = []
for text in combined:
    parsed_text = split_text(text)
    lemmatized_text = lemmatize(parsed_text, morph)
    n_gram_text = generate_n_grams(lemmatized_text, 5)
    uni_pos_text = generate_pos_unigrams_from_n_grams(n_gram_text, morph)
    pos_n_grams_text = get_pos_n_grams_string(uni_pos_text)
    pos_5_grams_dataset.append(pos_n_grams_text)
author_pos_5 = pd.DataFrame()
author_pos_5['pos'] = pos_5_grams_dataset
author_pos_5['author'] = labels
author_pos_5.to_csv('datasets/pos_5_30.csv', index=False)

pos_6_grams_dataset = []
for text in combined:
    parsed_text = split_text(text)
    lemmatized_text = lemmatize(parsed_text, morph)
    n_gram_text = generate_n_grams(lemmatized_text, 6)
    uni_pos_text = generate_pos_unigrams_from_n_grams(n_gram_text, morph)
    pos_n_grams_text = get_pos_n_grams_string(uni_pos_text)
    pos_6_grams_dataset.append(pos_n_grams_text)
author_pos_6 = pd.DataFrame()
author_pos_6['pos'] = pos_6_grams_dataset
author_pos_6['author'] = labels
author_pos_6.to_csv('datasets/pos_6_30.csv', index=False)

