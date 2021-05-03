from typing import List
from nltk import tokenize
import string
from pymorphy2 import MorphAnalyzer


def split_text(filepath: str, min_char: int = 10) -> List[List]:
    with open(filepath, 'r', encoding='utf8') as file:
        text = file.read().replace('\n', '. ')
        text = text.replace('.”', '”.').replace('."', '".').replace('?”', '”?').replace('!”', '”!')
        text = text.replace('--', ' ').replace('. . .', '').replace('_', '')

    sentences = tokenize.sent_tokenize(text)
    sentences = [sentence for sentence in sentences if len(sentence) >= min_char]

    normed_text = []

    for sentence in sentences:
        sentence = sentence.lower()
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        sentence = sentence.replace('“', '').replace('”', '')
        sentence = sentence.replace('‟', '').replace('”', '')
        sentence = sentence.replace('«', '').replace('»', '')
        sentence = sentence.replace('—', '').replace('–', '')
        sentence = sentence.replace('(', '').replace(')', '')
        sentence = sentence.replace('…', '')

        normed_text.append(sentence)

    morph = MorphAnalyzer()

    lemmatized_text = []

    for sent in normed_text:
        lemmatized_sentence = lemmatize(sent, morph)
        lemmatized_text.append(lemmatized_sentence)

    return list(lemmatized_text)


def lemmatize(sentence: str, morph: MorphAnalyzer) -> List:
    tokens = tokenize.word_tokenize(sentence)
    lemmas = []
    for token in tokens:
        lemma = morph.normal_forms(token)[0]
        lemmas.append(lemma)

    return lemmas
