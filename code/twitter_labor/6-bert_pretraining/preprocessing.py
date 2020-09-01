from tqdm import tqdm
from ftfy import fix_text
from html.parser import HTMLParser
import re
import string
import nltk
import argparse


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()

    # necessary
    parser.add_argument("--input_file_path", type=str, help="Path to the input data. Must be in txt format.")
    parser.add_argument("--text_language", type=str, help="Define language of the text. Possible options are german or portuguese")
    parser.add_argument("--lower_case", type=bool, help="Whether to lower case")
    parser.add_argument("--min_sentence_length", type=int)
    parser.add_argument("--blank_between_documents", type=bool)
    parser.add_argument("--delete_empty_lines", type=bool, default=False)
    args = parser.parse_args()
    return args


def string_contains_punctuation(string, punctuation_list):
    if any(punctuation in string for punctuation in punctuation_list):
        return True
    else:
        return False


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def text_cleaning(input_file_path):
    with open(input_file_path, "r+") as f:
        d = f.readlines()
        f.seek(0)
        print("Starting the text cleaning for {}".format(input_file_path))
        for line in tqdm(d):
            # links
            line = re.sub(r'(http|https|www)\S+', '', line)
            # email addresses
            line = re.sub(r'\S*@\S*\s?', '', line)
            # german phone numbers
            line = re.sub('(\(?([\d \-\)\–\+\/\(]+)\)?([ .\-–\/]?)([\d]+))', '', line)
            # fix encoding
            line = fix_text(line)
            # avoid common splitting mistake
            line = line.replace("ca.", "ca")
            if args.lower_case == True:
                line = line.lower()
            # HTML tags
            try:
                line = strip_tags(line)
            except NotImplementedError:
                pass
            
            if string_contains_punctuation(line, ".") and not string_contains_punctuation(line, "|"):
                f.write(line)
        f.truncate()


def sentence_splitting(input_file_path, sent_tokenizer, min_sentence_length):
    with open(input_file_path, "r+") as f:
        d = f.readlines()
        f.seek(0)
        print("Starting to split sentences for {}".format(input_file_path))
        for line in tqdm(d):
            new_line = ""
            for sent in sent_tokenizer.tokenize(line):
                if len(str(sent)) > min_sentence_length and string_contains_punctuation(str(sent), ".;"):
                    new_line = new_line + str(sent) + "\n"
            #add extra blank line between tweets
            f.write(new_line)
            if args.blank_between_documents == True:
                f.write('\n')
        f.truncate()


def delete_empty_lines(input_file_path):
    with open(input_file_path, "r+") as f:
        d = f.readlines()
        f.seek(0)
        print("Starting to delete empty lines for {}".format(input_file_path))
        for line in tqdm(d):
            if not line.strip(): continue  # skip the empty line
            f.write(line)
        f.truncate()


if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    nltk.download('punkt')
    sent_tokenizer = nltk.data.load('tokenizers/punkt/{}.pickle'.format(args.text_language))

    text_cleaning(args.input_file_path)
    sentence_splitting(args.input_file_path, sent_tokenizer, min_sentence_length=args.min_sentence_length)
    #if args.delete_empty_lines == True:
        #delete_empty_lines(args.input_file_path)
