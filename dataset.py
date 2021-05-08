import glob
import random
import numpy as np
import pandas as pd
from pymorphy2 import MorphAnalyzer

from parse_text import split_text, lemmatize_sentence, get_pos_sentence


chekhov = []
for path in glob.glob('./prose/Chekhov/*.txt'):
    chekhov += split_text(path)

dostoevsky = []
for path in glob.glob('./prose/Dostoevsky/*.txt'):
    dostoevsky += split_text(path)

tolstoy = []
for path in glob.glob('./prose/Tolstoy/*.txt'):
    tolstoy += split_text(path)

gogol = []
for path in glob.glob('./prose/Gogol/*.txt'):
    gogol += split_text(path)

gorky = []
for path in glob.glob('./prose/Gorky/*.txt'):
    gorky += split_text(path)

turgenev = []
for path in glob.glob('./prose/Gorky/*.txt'):
    turgenev += split_text(path)


corpus = dict.fromkeys(['chekhov', 'dostoevsky', 'tolstoy', 'gogol', 'gorky', 'turgenev'])

corpus['chekhov'] = chekhov
corpus['dostoevsky'] = dostoevsky
corpus['tolstoy'] = tolstoy
corpus['gogol'] = gogol
corpus['gorky'] = gorky
corpus['turgenev'] = turgenev


# np.random.seed(1)
#
# max_len = 10000
#
# names = [chekhov, dostoevsky, tolstoy, gogol, gorky, turgenev]
#
# combined = []
# for name in names:
#     name = np.random.choice(name, max_len, replace=False)
#     combined += list(name)
#
# labels = ['Chekhov'] * max_len + ['Dostoevsky'] * max_len + ['Tolstoy'] * max_len + ['Gogol'] * max_len \
#          + ['Gorky'] * max_len + ['Turgenev'] * max_len
#
# random.seed(3)
#
# zipped = list(zip(combined, labels))
# random.shuffle(zipped)
# combined, labels = zip(*zipped)
#
# author_dataset = pd.DataFrame()
# author_dataset['text'] = combined
# author_dataset['author'] = labels
#
# author_dataset.to_csv('author_dataset.csv', index=False)
#
# morph = MorphAnalyzer()
#
# pos_text = []
# for sent in combined:
#     sent = lemmatize_sentence(sent, morph)
#     sent = get_pos_sentence(sent, morph)
#     pos_text.append(sent)
#
# author_pos_dataset = pd.DataFrame()
# author_pos_dataset['text'] = pos_text
# author_pos_dataset['author'] = labels
#
# author_pos_dataset.to_csv('author_pos_dataset.csv', index=False)
