from parse_text import split_text


def main():
    parsed_text = split_text('./text.txt')
    print(parsed_text)


if __name__ == '__main__':
    main()