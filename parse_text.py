from typing import List
from nltk import tokenize
import string


def split_text(filepath: str, morph, min_char: int = 10) -> List[str]:
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

    return list(normed_text)


def lemmatize(text: List[str], morph) -> List[List]:
    lemmatized_text = []
    for sentence in text:
        tokens = tokenize.word_tokenize(sentence)
        lemmas = []
        for token in tokens:
            lemma = morph.normal_forms(token)[0]
            lemmas.append(lemma)
        lemmatized_text.append(lemmas)

    return lemmatized_text
