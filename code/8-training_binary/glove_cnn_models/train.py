#! /usr/bin/env python

import os
import time
import datetime
import sys
import shutil
import glob

import data_utils as utils
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

from data_utils import IMDBDataset
from text_cnn import TextCNN

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
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("preprocessing", False, "Whether to preprocess tweets or not")
# Specifics
tf.flags.DEFINE_string("data_path",
                       "/home/manuto/Documents/world_bank/bert_twitter_labor/code/twitter/data/may20_9Klabels/data_binary_pos_neg_balanced",
                       "path to train and val data")
tf.flags.DEFINE_string("embeddings_path",
                       "/home/manuto/Documents/world_bank/bert_twitter_labor/data/glove_embeddings/embeddings.npy",
                       "path to embeddings npy file")
tf.flags.DEFINE_string("label", "is_unemployed", "Label to train on")
tf.flags.DEFINE_string("vocab_path",
                       "/home/manuto/Documents/world_bank/bert_twitter_labor/data/glove_embeddings/vocab.pckl",
                       "Path pickle file")
tf.flags.DEFINE_string("run_name", "default_run_name", "Name of folder in runs folder where models are saved")
tf.flags.DEFINE_string("output_dir", "", "Output directory where models are saved")


FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
print("Loading Dataset ...")

data_path = FLAGS.data_path
train_df = pd.read_csv(os.path.join(data_path, "train_{}.csv".format(FLAGS.label)), lineterminator = '\n')
eval_df = pd.read_csv(os.path.join(data_path, "val_{}.csv".format(FLAGS.label)), lineterminator= '\n')



def tokenizer(text):
    return [wdict.get(w.lower(), 0) for w in text.split(' ')]


with open(FLAGS.vocab_path, 'rb') as dfile:
    wdict = pickle.load(dfile)

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'le npercent', 'money', 'phone', 'user',
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

if FLAGS.preprocessing:
    train_df['text'] = train_df['text'].apply(ekphrasis_preprocessing)
    eval_df['text'] = eval_df['text'].apply(ekphrasis_preprocessing)
    print("***********Text was successfully preprocessed***********")


train_df['text_tokenized'] = train_df['text'].apply(tokenizer)
eval_df['text_tokenized'] = eval_df['text'].apply(tokenizer)


def pad_dataset(dataset, maxlen):
    return np.array(
        [np.pad(r, (0, maxlen - len(r)), mode='constant') if len(r) < maxlen else np.array(r[:maxlen])
         for r in dataset])


x_train = pad_dataset(train_df.text_tokenized.values.tolist(), 128)
x_dev = pad_dataset(eval_df.text_tokenized.values.tolist(), 128)


def create_label(label):
    if label == 1:
        return [0, 1]
    elif label == 0:
        return [1, 0]


y_train = np.array((train_df['class'].apply(create_label)).values.tolist())
y_dev = np.array((eval_df['class'].apply(create_label)).values.tolist())

vocab_size = len(wdict)
embedding_path = FLAGS.embeddings_path
embedding = utils.load_embeddings(embedding_path, vocab_size, FLAGS.embedding_dim)
print("Embeddings loaded, Vocabulary Size: {:d}. Starting training ...".format(vocab_size))

def prepare_filepath_for_storing_model(output_dir: str) -> str:
    """Prepare the filepath where the trained model will be stored.

    :param output_dir: Directory where to store outputs (trained models).
    :return: path_to_store_model: Path where to store the trained model.
    """
    path_to_store_model = os.path.join(output_dir, 'models')
    if not os.path.exists(path_to_store_model):
        os.makedirs(path_to_store_model)
    return path_to_store_model

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
        out_dir = prepare_filepath_for_storing_model(output_dir=args.output_dir)
        #out_dir = os.path.abspath(os.path.join(FLAGS.output_dir, FLAGS.run_name))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss, accuracy, precision
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        precision_summary = tf.summary.scalar("precision", cnn.precision)
        recall_summary = tf.summary.scalar("recall", cnn.recall)
        auc_summary = tf.summary.scalar("auc", cnn.auc)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged, precision_summary, recall_summary, auc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary, precision_summary, recall_summary, auc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
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
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall, cnn.auc], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, precision {:g}, recall {:g}, auc {:g}".format(time_str, step, loss, accuracy, precision, recall, auc))
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
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, precision {:g}, recall {:g}, auc {:g}".format(time_str, step, loss, accuracy, precision, recall, auc))
            if writer:
                writer.add_summary(summaries, step)
            return loss


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

            #if current_step % FLAGS.checkpoint_every == 0:
            #    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            #    print("Saved model checkpoint to {}\n".format(path))
