from typing import List
from nltk import tokenize
import string


def split_text(filepath: str, min_char: int = 10) -> list[str]:
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


def main():
    tokenized_text = split_text('./text.txt')
    print(tokenized_text)

    
if __name__ == '__main__':
    main()