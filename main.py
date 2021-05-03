from pymorphy2 import MorphAnalyzer

from n_grams import generate_n_grams, generate_pos_n_grams
from parse_text import split_text


def main():
    morph = MorphAnalyzer()
    parsed_text = split_text('./text.txt', morph)
    n_grams_array = generate_n_grams(parsed_text, 5)
    generate_pos_n_grams(n_grams_array, morph)


if __name__ == '__main__':
    main()