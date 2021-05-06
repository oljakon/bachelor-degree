import glob
import random
import numpy as np
import pandas as pd

from parse_text import split_text


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

np.random.seed(1)

max_len = 10000

names = [chekhov, dostoevsky, tolstoy, gogol, gorky, turgenev]

combined = []
for name in names:
    name = np.random.choice(name, max_len, replace=False)
    combined += list(name)

labels = ['Chekhov'] * max_len + ['Dostoevsky'] * max_len + ['Tolstoy'] * max_len + ['Gogol'] * max_len \
         + ['Gorky'] * max_len + ['Turgenev'] * max_len

random.seed(3)

zipped = list(zip(combined, labels))
random.shuffle(zipped)
combined, labels = zip(*zipped)

author_dataset = pd.DataFrame()
author_dataset['text'] = combined
author_dataset['author'] = labels

author_dataset.to_csv('author_dataset.csv', index=False)