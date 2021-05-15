from typing import List
from nltk import tokenize
import string


def read_text_from_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf8') as file:
        text = file.read().replace('\n', '. ')

    return text


def split_text(text: str, min_char: int = 20) -> List[str]:
    text = text.replace('.”', '”.').replace('."', '".').replace('?”', '”?').replace('!”', '”!')
    text = text.replace('--', ' ').replace('. . .', '').replace('_', '')
    text = text.replace('\xa0', '')

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

    return list(normed_text)


def lemmatize(text: List[str], morph) -> List[str]:
    lemmatized_text = []
    for sentence in text:
        tokens = tokenize.word_tokenize(sentence)
        lemmas = ''
        for token in tokens:
            lemma = morph.normal_forms(token)[0]
            lemmas += lemma + ' '
        lemmas = lemmas.rstrip()
        lemmatized_text.append(lemmas)

    return lemmatized_text


def lemmatize_sentence(sentence: str, morph) -> str:
    lemmas_sentence = ''
    tokens = tokenize.word_tokenize(sentence)
    for token in tokens:
        lemma = morph.normal_forms(token)[0]
        lemmas_sentence += lemma + ' '
    lemmas_sentence = lemmas_sentence.rstrip()

    return lemmas_sentence


def get_pos_sentence(sentence: str, morph) -> str:
    pos_sentence = ''
    tokens = tokenize.word_tokenize(sentence)
    for token in tokens:
        pos = morph.parse(token)[0].tag.POS
        pos_sentence += str(pos) + ' '
    pos_sentence = pos_sentence.rstrip()

    return pos_sentence
