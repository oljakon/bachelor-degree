from n_grams import generate_n_grams
from parse_text import split_text


def main():
    parsed_text = split_text('./text.txt')
    n_grams_array = generate_n_grams(parsed_text, 3)


if __name__ == '__main__':
    main()