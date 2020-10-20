import pandas as pd
import argparse
import re


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str)
    args = parser.parse_args()
    return args


def clean_text(text):
    text = text.lower()
    accent_replacements = [
        ('á|à|ã', 'a'),
        ('é|ê|è', 'e'),
        ('í', 'i'),
        ('ò|ó|õ', 'o'),
        ('ú|ü', 'u'),
        ('ñ', 'n'),
        ('ç', 'c')]
    for a, b in accent_replacements:
        text = re.sub(a, b, text)
    text = text.replace(':', '')
    text = text.replace(',', '')
    text = text.replace('"', '')
    text = text.replace(' ', '_')
    return text


def normalize_unicode(unicode):
    char_list = unicode.split(' ')
    i = 0
    for c in char_list:
        if len(c) == 6:
            c = c.replace('+', '0000')
        else:
            c = c.replace('+', '000')
        char_list[i] = c
        i = i + 1
    code = ''.join(char_list)
    char = "u'" + code.replace('U', '\\U') + "',"
    return char


if __name__ == "__main__":
    args = get_args_from_command_line()
    folder_path = '/home/manuto/Documents/world_bank/bert_twitter_labor/code/twitter/code/2-twitter_labor/6-bert_pretraining/emoji'
    csv_path = f'{folder_path}/emoji_{args.language}_raw.csv'
    df = pd.read_csv(csv_path)
    df.columns = ['nb', 'emoji', 'emoji_text', 'unicode']
    # normalize text
    df['clean_text'] = df['emoji_text'].apply(clean_text)
    # normalize unicode
    df['normalize_unicode'] = df['unicode'].apply(normalize_unicode)
    df = df[['clean_text', 'normalize_unicode']]
    df.set_index('clean_text', inplace=True)
    output_dict = df.to_dict()
    for name in sorted(output_dict.keys()):
        print("    u':%s:': %s" % (name, output_dict[name]))
