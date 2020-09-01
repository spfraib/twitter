import os
import logging
import argparse
from simpletransformers.classification import ClassificationModel
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sklearn
from scipy.special import softmax
import numpy as np
from datetime import datetime
import time
import pytz

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_output_folder", type=str, help="Path to the inference data. Must be in csv format.",
                        default="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Define args from command line
    args = get_args_from_command_line()
    # Import inference data
    inference_df = pd.read_csv(args.inference_data_path)