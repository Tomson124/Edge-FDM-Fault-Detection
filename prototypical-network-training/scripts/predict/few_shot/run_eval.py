"""
This code was adapted from the original implementation of Prototypical Networks for Few-Shot Learning.
Link: https://github.com/jakesnell/prototypical-networks
"""

import argparse
import os
from eval import main

parser = argparse.ArgumentParser(description='Evaluate few-shot prototypical networks')

default_model_path = 'results/std-cnn/best_model.pt'
default_dataset = os.path.join(os.path.dirname(__file__), '../../../datasets/combined_ds')
parser.add_argument('--model.model_path', type=str, default=default_model_path, metavar='MODELPATH',
                    help="location of pretrained model to evaluate (default: {:s})".format(default_model_path))
model_options_path = 'results/std-cnn/opt.json'
parser.add_argument('--model.model_options_path', type=str, default=model_options_path, metavar='MODELOPTIONS',
                    help="location of model options (default: {:s})".format(model_options_path))

parser.add_argument('--data.test_way', type=int, default=2, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as model's data.test_way (default: 0)")
parser.add_argument('--data.test_shot', type=int, default=5, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as model's data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=15, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as model's data.query (default: 0)")
parser.add_argument('--data.test_episodes', type=int, default=1000, metavar='NTEST',
                    help="number of test episodes per epoch (default: 1000)")
parser.add_argument('--data.dataset', type=str, default=default_dataset, metavar='DS')
parser.add_argument('--f1', action='store_true', help="Compute F1 score")

args = vars(parser.parse_args())
main(args)
