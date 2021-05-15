import glob
import random
import numpy as np
import pandas as pd

from parse_text import read_text_from_file

chekhov = []
for path in glob.glob('./prose/Chekhov/*.txt'):
    chekhov.append(read_text_from_file(path))

dostoevsky = []
for path in glob.glob('./prose/Dostoevsky/*.txt'):
    dostoevsky.append(read_text_from_file(path))

tolstoy = []
for path in glob.glob('./prose/Tolstoy/*.txt'):
    tolstoy.append(read_text_from_file(path))

gogol = []
for path in glob.glob('./prose/Gogol/*.txt'):
    gogol.append(read_text_from_file(path))

gorky = []
for path in glob.glob('./prose/Gorky/*.txt'):
    gorky.append(read_text_from_file(path))

turgenev = []
for path in glob.glob('./prose/Gorky/*.txt'):
    turgenev.append(read_text_from_file(path))

max_len = 15

chekhov = chekhov[:max_len]
dostoevsky = dostoevsky[:max_len]
tolstoy = tolstoy[:max_len]
gogol = gogol[:max_len]
gorky = gorky[:max_len]
turgenev = turgenev[:max_len]

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

author_text = pd.DataFrame()
author_text['text'] = combined
author_text['author'] = labels

author_text.to_csv('author_text.csv', index=False)
