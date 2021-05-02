import glob
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

text_dict = {'Chekhov': chekhov, 'Dostoevsky': dostoevsky, 'Tolstoy': tolstoy}

# for key in text_dict.keys():
#     print(key, ':', len(text_dict[key]), ' sentences')
