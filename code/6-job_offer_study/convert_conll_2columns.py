import argparse

# Tag mapping
mapping = {"I-LOC":"LOC", "B-LOC":"LOC", "I-ORG":"ORG", "B-ORG":"ORG"}


parser = argparse.ArgumentParser(description='Script to convert CoNLL 2003 dataset to the Job offer tags using mapping dictionary, Output format: 2 columns(word, tag)')
parser.add_argument("--input_file",help="Input dataset file", required=True)
parser.add_argument("--out_file", help="Output file", required=True)


# Parse args
args = parser.parse_args()
conll_data = args.input_file
converted_conll_data = args.out_file


def main():
    with open(conll_data) as f, open(converted_conll_data,'w') as f1:
        data = f.readlines()
        for d in data:
            d1 = d.split(" ")
            label = d1[-1].replace("\n","")
            mapped_label = None
            if label in mapping:
                mapped_label = mapping[label]
            else:
                mapped_label = "MISC"
            
            if(len(d1)<2):
                f1.write(d1[0])
            else:
                f1.write(" ".join([d1[0],mapped_label]))
                f1.write("\n")
    

if __name__ == "__main__":
    main()