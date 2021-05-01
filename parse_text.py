from typing import List
from nltk import tokenize, download


def split_text(filepath: str, min_char: int = 5) -> list[str]:
    with open(filepath, 'r', encoding='utf8') as file:
        text = file.read().replace('\n', '. ')
        text = text.replace('.”', '”.').replace('."', '".').replace('?”', '”?').replace('!”', '”!')
        text = text.replace('--', ' ').replace('. . .', '').replace('_', '')

    sentences = tokenize.sent_tokenize(text)
    sentences = [sentence for sentence in sentences if len(sentence) >= min_char]

    return list(sentences)


def main():
    tokenized_text = split_text('/Users/olga/Documents/bachelor-degree/text.txt')
    print(tokenized_text)

    
if __name__ == '__main__':
    main()