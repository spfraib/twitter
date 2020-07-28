#! /usr/bin/env python

import os
import time
import datetime
import sys
import shutil
import glob
from datetime import datetime
import pytz
import data_utils as utils

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

from text_cnn import TextCNN
from sklearn import metrics

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (300 for this example)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 40, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("preprocessing", True, "Whether to preprocess tweets or not")
# Specifics
tf.flags.DEFINE_string("training_data_path",
                       "may20_9Klabels/data_binary_pos_neg_balanced",
                       "path to train and val data")
tf.flags.DEFINE_string("holdout_data_path",
                       '/home/manuto/Documents/world_bank/bert_twitter_labor/twitter-labor-data/data/jul23_iter0/preprocessed',
                       "path to holdout data")
tf.flags.DEFINE_string("embeddings_path",
                       "/home/manuto/Documents/world_bank/bert_twitter_labor/data/glove_embeddings/embeddings.npy",
                       "path to embeddings npy file")
tf.flags.DEFINE_string("label", "is_unemployed", "Label to train on")
tf.flags.DEFINE_string("vocab_path",
                       "/home/manuto/Documents/world_bank/bert_twitter_labor/data/glove_embeddings/vocab.pckl",
                       "Path pickle file")
tf.flags.DEFINE_string("output_dir",
                       "/home/manuto/Documents/world_bank/bert_twitter_labor/data/trained_glove_models/GloVe_CNN_<DATA_PATH>_<JOB_ID>",
                       "Output directory where models are saved")
tf.flags.DEFINE_string("slurm_job_timestamp", "1595779138", "Timestamp when job is launched")
tf.flags.DEFINE_string("slurm_job_id", "0", "ID of the job that ran training")


# UTILS

def tokenizer(text):
    return [wdict.get(w.lower(), 0) for w in text.split(' ')]


text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


def ekphrasis_preprocessing(tweet):
    return " ".join(text_processor.pre_process_doc(tweet))


def pad_dataset(dataset, maxlen):
    return np.array(
        [np.pad(r, (0, maxlen - len(r)), mode='constant') if len(r) < maxlen else np.array(r[:maxlen])
         for r in dataset])


def create_label(label):
    if label == 1:
        return [0, 1]
    elif label == 0:
        return [1, 0]


def prepare_filepath_for_storing_model(output_dir: str) -> str:
    """Prepare the filepath where the trained model will be stored.

    :param output_dir: Directory where to store outputs (trained models).
    :return: path_to_store_model: Path where to store the trained model.
    """
    path_to_store_model = os.path.join(output_dir, 'models')
    if not os.path.exists(path_to_store_model):
        os.makedirs(path_to_store_model)
    return path_to_store_model


def train_step(x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, summaries, loss, accuracy, precision, recall, auc = sess.run(
        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall, cnn.auc],
        feed_dict)
    time_str = datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}, precision {:g}, recall {:g}, auc {:g}".format(time_str, step, loss,
                                                                                           accuracy, precision,
                                                                                           recall, auc))
    train_summary_writer.add_summary(summaries, step)


def dev_step(x_batch, y_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: 1.0
    }
    step, summaries, loss, accuracy, precision, recall, auc = sess.run(
        [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall, cnn.auc], feed_dict)
    time_str = datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}, precision {:g}, recall {:g}, auc {:g}".format(time_str, step, loss,
                                                                                           accuracy, precision,
                                                                                           recall, auc))
    if writer:
        writer.add_summary(summaries, step)
    return loss


# Load arguments
FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

# Data Preparation
print("Loading Dataset ...")

training_data_path = os.path.join('/scratch/mt4493/twitter_labor/twitter-labor-data/data/',
                                  FLAGS.training_data_path)
if '/' in FLAGS.training_data_path:
    training_data_path_no_slash = FLAGS.training_data_path.replace('/', '_')

train_df = pd.read_csv(os.path.join(training_data_path, "train_{}.csv".format(FLAGS.label)))  # , lineterminator='\n')
eval_df = pd.read_csv(os.path.join(training_data_path, "val_{}.csv".format(FLAGS.label)))  # , lineterminator='\n')
if FLAGS.holdout_data_path:
    holdout_df = pd.read_csv(os.path.join(FLAGS.holdout_data_path, "holdout_{}.csv".format(FLAGS.label)),
                             lineterminator='\n')
# Load vocabulary
with open(FLAGS.vocab_path, 'rb') as dfile:
    wdict = pickle.load(dfile)

# Perform preprocessing
if FLAGS.preprocessing:
    train_df['text'] = train_df['text'].apply(ekphrasis_preprocessing)
    eval_df['text'] = eval_df['text'].apply(ekphrasis_preprocessing)
    if FLAGS.holdout_data_path:
        holdout_df['text'] = holdout_df['text'].apply(ekphrasis_preprocessing)
    print("***********Text was successfully preprocessed***********")

# Tokenize and prepare input data
train_df['text_tokenized'] = train_df['text'].apply(tokenizer)
eval_df['text_tokenized'] = eval_df['text'].apply(tokenizer)

x_train = pad_dataset(train_df.text_tokenized.values.tolist(), 128)
x_dev = pad_dataset(eval_df.text_tokenized.values.tolist(), 128)

y_train = np.array((train_df['class'].apply(create_label)).values.tolist())
y_dev = np.array((eval_df['class'].apply(create_label)).values.tolist())

if FLAGS.holdout_data_path:
    holdout_df['text_tokenized'] = holdout_df['text'].apply(tokenizer)
    x_holdout = pad_dataset(holdout_df.text_tokenized.values.tolist(), 128)
    y_holdout = np.array((holdout_df['class'].apply(create_label)).values.tolist())

vocab_size = len(wdict)
embedding_path = FLAGS.embeddings_path
embedding = utils.load_embeddings(embedding_path, vocab_size, FLAGS.embedding_dim)
print("Embeddings loaded, Vocabulary Size: {:d}. Starting training ...".format(vocab_size))

# Training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=vocab_size,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = FLAGS.output_dir
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss, accuracy, precision
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        precision_summary = tf.summary.scalar("precision", cnn.precision)
        recall_summary = tf.summary.scalar("recall", cnn.recall)
        auc_summary = tf.summary.scalar("auc", cnn.auc)

        # Train Summaries
        train_summary_op = tf.summary.merge(
            [loss_summary, acc_summary, grad_summaries_merged, precision_summary, recall_summary, auc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary, precision_summary, recall_summary, auc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'models', "best_model"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        # vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(cnn.embedding_init, feed_dict={cnn.embedding_placeholder: embedding})

        # Generate batches
        batches = utils.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        lowest_eval_loss = 1
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                loss = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
                if loss < lowest_eval_loss:
                    lowest_eval_loss = loss
                    checkpoint_folder = glob.glob(checkpoint_dir + '/*')
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved best model checkpoint to {}\n".format(path))
                    for f in checkpoint_folder:
                        head, tail = os.path.split(f)
                        if tail != 'checkpoint':
                            os.remove(f)
                    print("Removed former best model")

            # if current_step % FLAGS.checkpoint_every == 0:
            #    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            #    print("Saved model checkpoint to {}\n".format(path))

# Evaluation
print('*************Training is done. Starting evaluation*************')
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        predictions_proba = graph.get_operation_by_name("output/predictions_proba").outputs[0]
        # predictions_proba = predictions_proba[:, 1]
        # Generate batches for one epoch
        batches_dev = utils.batch_iter(list(x_dev), FLAGS.batch_size, 1, shuffle=False)
        if FLAGS.holdout_data_path:
            batches_holdout = utils.batch_iter(list(x_holdout), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions_eval = []
        all_predictions_proba_eval = []
        all_predictions_holdout = []
        all_predictions_proba_holdout = []
        for x_dev_batch in batches_dev:
            batch_predictions = sess.run(predictions, {input_x: x_dev_batch, dropout_keep_prob: 1.0})
            batch_predictions_proba = sess.run(predictions_proba, {input_x: x_dev_batch, dropout_keep_prob: 1.0})
            all_predictions_eval = np.concatenate([all_predictions_eval, batch_predictions])
            all_predictions_proba_eval = np.concatenate([all_predictions_proba_eval, batch_predictions_proba[:, 1]])
        if FLAGS.holdout_data_path:
            for x_holdout_batch in batches_holdout:
                batch_predictions = sess.run(predictions, {input_x: x_holdout_batch, dropout_keep_prob: 1.0})
                batch_predictions_proba = sess.run(predictions_proba,
                                                   {input_x: x_holdout_batch, dropout_keep_prob: 1.0})
                all_predictions_holdout = np.concatenate([all_predictions_holdout, batch_predictions])
                all_predictions_proba_holdout = np.concatenate(
                    [all_predictions_proba_holdout, batch_predictions_proba[:, 1]])

# Save eval set results
if all_predictions_eval is not None:
    print("Predictions on evaluation set done")
    print("Total number of evaluation examples in evaluation set: {}".format(len(all_predictions_proba_eval)))
    eval_df['glove_cnn_scores'] = all_predictions_proba_eval

if FLAGS.holdout_data_path:
    if all_predictions_holdout is not None:
        print("Predictions on holdout set done")
        print("Total number of evaluation examples in holdout set: {}".format(len(all_predictions_proba_holdout)))
        holdout_df['glove_cnn_scores'] = all_predictions_proba_holdout

# Compute AUC
fpr, tpr, thresholds = metrics.roc_curve(eval_df['class'], all_predictions_proba_eval)
auc_eval = metrics.auc(fpr, tpr)
# Build final results dictionary
eval_results_eval_set_dict = {'slurm_job_id': FLAGS.slurm_job_id,
                              'slurm_job_timestamp': FLAGS.slurm_job_timestamp,
                              'slurm_job_Berlin_date_time': str(datetime.fromtimestamp(int(FLAGS.slurm_job_timestamp),
                                                                                       tz=pytz.timezone(
                                                                                           'Europe/Berlin'))),
                              'model_type': 'GloVe + CNN',
                              'evaluation_data_path': os.path.join(training_data_path,
                                                                   'val_{}.csv'.format(FLAGS.label)),
                              'precision': metrics.precision_score(eval_df['class'], all_predictions_eval),
                              'recall': metrics.recall_score(eval_df['class'], all_predictions_eval),
                              'f1': metrics.f1_score(eval_df['class'], all_predictions_eval),
                              'auc': auc_eval
                              }
# Save eval set results in CSV
name_val_file = 'val_{}'.format(FLAGS.label)
path_to_store_eval_results = os.path.join(training_data_path, 'results',
                                          'GloVe_CNN_' + str(FLAGS.slurm_job_id),
                                          name_val_file + '_evaluation.csv')
if not os.path.exists(os.path.dirname(path_to_store_eval_results)):
    os.makedirs(os.path.dirname(path_to_store_eval_results))
pd.DataFrame.from_dict(eval_results_eval_set_dict, orient='index', columns=['value']).to_csv(
    path_to_store_eval_results)
print("The evaluation on the evaluation set is done. The results were saved at {}".format(path_to_store_eval_results))
# Save eval scores to CSV
path_to_store_eval_scores = os.path.join(training_data_path, 'results', 'GloVe_CNN_' + str(FLAGS.slurm_job_id),
                                         name_val_file + '_scores.csv')

eval_df.to_csv(path_to_store_eval_scores, index=False)
print("The scores for the evaluation set were saved at {}".format(path_to_store_eval_scores))

if FLAGS.holdout_data_path:
    # Compute AUC
    fpr, tpr, thresholds = metrics.roc_curve(holdout_df['class'], all_predictions_proba_holdout)
    auc_holdout = metrics.auc(fpr, tpr)
    # Build final results dictionary
    eval_results_holdout_set_dict = {'slurm_job_id': FLAGS.slurm_job_id,
                                     'slurm_job_timestamp': FLAGS.slurm_job_timestamp,
                                     'slurm_job_Berlin_date_time': str(
                                         datetime.fromtimestamp(int(FLAGS.slurm_job_timestamp),
                                                                tz=pytz.timezone(
                                                                    'Europe/Berlin'))),
                                     'model_type': 'GloVe + CNN',
                                     'holdout_data_path': os.path.join(FLAGS.holdout_data_path,
                                                                       'holdout_{}.csv'.format(FLAGS.label)),
                                     'precision': metrics.precision_score(holdout_df['class'], all_predictions_holdout),
                                     'recall': metrics.recall_score(holdout_df['class'], all_predictions_holdout),
                                     'f1': metrics.f1_score(holdout_df['class'], all_predictions_holdout),
                                     'auc': auc_holdout
                                     }
    # Save eval set results in CSV
    name_holdout_file = 'holdout_{}'.format(FLAGS.label)
    path_to_store_holdout_results = os.path.join(training_data_path, 'results',
                                                 'GloVe_CNN_' + str(FLAGS.slurm_job_id),
                                                 name_holdout_file + '_evaluation.csv')
    if not os.path.exists(os.path.dirname(path_to_store_holdout_results)):
        os.makedirs(os.path.dirname(path_to_store_holdout_results))
    pd.DataFrame.from_dict(eval_results_holdout_set_dict, orient='index', columns=['value']).to_csv(
        path_to_store_holdout_results)
    print(
        "The evaluation on the holdout set is done. The results were saved at {}".format(path_to_store_holdout_results))
    # Save eval scores to CSV
    path_to_store_holdout_scores = os.path.join(training_data_path, 'results', 'GloVe_CNN_' + str(FLAGS.slurm_job_id),
                                                name_holdout_file + '_scores.csv')

    holdout_df.to_csv(path_to_store_holdout_scores, index=False)
    print("The scores for the holdout set were saved at {}".format(path_to_store_holdout_scores))
