from simpletransformers.ner.ner_model import NERModel
import csv 
import argparse

parser = argparse.ArgumentParser("Train NER model")
parser.add_argument("--CoNLL_2003_data", help = "training set for CoNLL 2003 data converted to twitter format", default = "data/conll_train.txt")
parser.add_arguments("--output_dir_ner", help = "output directory to save model trained on CoNLL dataset", default = "outputs/CoNLL")
parser.add_argument("--job_offer_train_set", help = "data/job_offer_train.txt")
parser.add_argument("--job_offer_valid_set", help = "data/job_offer_valid.txt")
parser.add_arguments("--output_dir_job_offer", help = "output directory to save model trained on job offer dataset", default = "outputs/job_offer")


# Parse arguments

args = parser.parse_args()
conll_train_set = args.CoNLL_2003_data
conll_output_dir = args.output_ner
job_offer_train_set = args.job_offer_train_set
job_offer_valid_set = args.job_offer_valid_set
job_offer_output_dir = args.output_dir_job_offer

# Model args
model_args = {
    "output_dir": "outputs/",
    "cache_dir": "cache_dir/",
    "fp16": True,
    "fp16_opt_level": "O1",
     "max_seq_length": 128,
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "eval_batch_size": 8,
    "num_train_epochs": 5,
    "weight_decay": 0,
    "learning_rate": 4e-5,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "logging_steps": 50,
    "save_steps": 2000,
    "overwrite_output_dir": False,
    "reprocess_input_data": False,
    "evaluate_during_training": False,
    "n_gpu": 1,
}

LABELS = ["ORG","LOC", "JOB_TITLE", "MISC"]


def get_true_labels(filepath):
	f = open(filepath,'r',encoding="utf-8").readlines()
	sentences = []
	prev = -1
	for i,v in enumerate(f):
    		if v=="\n":
        		sentences.append(f[prev+1:i])
        		prev = i
	sentences.append(f[prev+1:])
	preds = []
	for sentence in sentences:
		labels = []
		for token in sentence:
			label = token.split(" ")[1][:-1]
			labels.append(label)
		preds.append(labels)
	return preds

def print_metrics(metrics):
	for metric in metrics:
		recall = metrics[metric]["TP"]/(metrics[metric]["TP"] + metrics[metric]["FN"])
		precision = metrics[metric]["TP"]/(metrics[metric]["TP"] + metrics[metric]["FP"])
		f1 = 2*recall*precision/(recall+precision)
		print(f"Metric: {metric}, Precision: {precision}, Recall: {recall}, F1: {f1}")	


# Calculate metrics
def calculate_metrics(predictions, actual_labels):
    metrics = {}

    for label in LABELS:
        metrics[label] = {}
        metrics[label]["TP"] = 0.0
        metrics[label]["FP"] = 0.0
        metrics[label]["FN"] = 0.0
        metrics[label]["TN"] = 0.0

    for i,prediction in enumerate(predictions):
        for j,label in enumerate(prediction):
            if label == actual_labels[i][j]:
                metrics[label]["TP"] += 1
            else:
                metrics[actual_labels[i][j]]["FN"] += 1
                metrics[label]["FP"] += 1
    return metrics

def main():
    # Create a NERModel
    model = NERModel('bert', 'bert-base-cased', args=model_args, labels = LABELS)

    # Train on CoNLL data

    model.train_model(conll_train_set,output_dir = conll_output_dir)

    # Train on job offer data
    model.train_model(job_offer_train_set,output_dir = job_offer_output_dir)

    # Evaluate the model
    result, model_outputs, predictions = model.eval_model(job_offer_valid_set)

    # Get actual validation labels
    actual_labels = get_true_labels(job_offer_valid_set)

    # Calculate metrics
    metrics = calculate_metrics(predictions, actual_labels)

    # print metrics
    print_metrics(metrics)


if __name__ == "__main__":
    main()