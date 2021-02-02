from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn import metrics
from argparse import Namespace
from pathlib import Path
import pandas as pd
import pyarrow
import os
import matplotlib.pyplot as plt
from collections import Counter
from nltk import ngrams
import string
import re
import argparse

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_folder", type=str,
                        default="iter_0-convbert-test-48-10-900-1538433-new_samples")
    parser.add_argument("--random_sample", type=str,
                        help="Random sample type.",
                        default="new_samples")
    parser.add_argument("--country_code", type=str,
                        default="US")
    parser.add_argument("--data_folder", type=str,
                        default="jan5_iter0")
    args = parser.parse_args()
    return args