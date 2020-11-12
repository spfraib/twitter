import csv
import argparse

parser = argparse.ArgumentParser(description='Script to convert dataset to NER')
parser.add_argument("--input_file",help="Input dataset file(csv)", required=True)
parser.add_argument("--out_file", help="Output file in NER format", required=True)

LABELS = ["ORG","LOC", "JOB_TITLE", "MISC"]



args = parser.parse_args()
job_offer_file = args.input_file
ner_output = args.out_file


output_file = open(ner_output,"w")


def write_to_file(job_offer_file, delimit = ',', STOP = None):
    # STOP is used to stop the processing at a particular line in the input file
    prev = None
    count = 0
    with open(job_offer_file) as csvfile:
        reader = csv.reader(csvfile, delimiter = delimit)
        next(reader,None)
        for i,row in enumerate(reader):
            token, actual_labels = row[-5], row[-4:-1]
            flag = 0
            if i!=0 and row[0]!=prev:
                output_file.write("\n")
                count += 1
            prev = row[0]
            for ind, val in enumerate(actual_labels):
                val = float(val)
                if val == 1:
                    flag = 1
                    output_file.write(token +" "+LABELS[ind]+"\n")
                    break 
            if flag==0:
                output_file.write(token +" "+LABELS[-1]+"\n")
        
            if STOP != None and i == STOP:
                 break

    output_file.write("\n")        


def main():
    write_to_file(job_offer_file)
    

if __name__ == "__main__":
    main()
        