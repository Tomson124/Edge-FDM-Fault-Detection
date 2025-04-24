"""
This code was adapted from the original implementation of Prototypical Networks for Few-Shot Learning.
Link: https://github.com/jakesnell/prototypical-networks
"""

import argparse
import json
from train import main

parser = argparse.ArgumentParser(description='Train prototypical networks')

default_dataset = '[DATASET_PATH_HERE]'
num_classes = 2
parser.add_argument('--data.dataset', type=str, default=default_dataset, metavar='DS',
                    help="data set name (default: {:s})".format(default_dataset))
default_split = 'vinyals'
parser.add_argument('--data.split', type=str, default=default_split, metavar='SP',
                    help="split name (default: {:s})".format(default_split))
parser.add_argument('--data.way', type=int, default=num_classes, metavar='WAY',
                    help="number of classes per episode (default: 2)")
parser.add_argument('--data.shot', type=int, default=5, metavar='SHOT',
                    help="number of support examples per class (default: 5)")
parser.add_argument('--data.query', type=int, default=5, metavar='QUERY',
                    help="number of query examples per class (default: 5)")
parser.add_argument('--data.test_way', type=int, default=num_classes, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as data.way (default: 5)")
parser.add_argument('--data.test_shot', type=int, default=0, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=15, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as data.query (default: 15)")
parser.add_argument('--data.train_episodes', type=int, default=100, metavar='NTRAIN',
                    help="number of train episodes per epoch (default: 100)")
parser.add_argument('--data.test_episodes', type=int, default=100, metavar='NTEST',
                    help="number of test episodes per epoch (default: 100)")
parser.add_argument('--data.trainval', action='store_true', help="run in train+validation mode (default: False)")
parser.add_argument('--data.sequential', action='store_true', help="use sequential sampler instead of episodic (default: False)")
parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")
parser.add_argument('--data.mps', action='store_true', help="run in MPS mode (default: False)")

default_model_name = 'protonet_conv'
parser.add_argument('--model.model_name', type=str, default=default_model_name, metavar='MODELNAME',
                    help="model name (default: {:s})".format(default_model_name))
parser.add_argument('--model.x_dim', type=str, default='3,224,224', metavar='XDIM',
                    help="dimensionality of input images (default: '3,224,224')")

parser.add_argument('--model.dropout', action=argparse.BooleanOptionalAction, default=False,
                    help="Enable or disable dropout (default: False)")
default_dropout_prob = {"fc": 0.0, "conv5": 0.0, "stages": {"stage3": 0.0, "stage4": 0.0}}
parser.add_argument('--model.dropout_prob', type=str, default=default_dropout_prob, metavar='DROPOUT_PROB',
                    help='dropout probabilities')
parser.add_argument('--model.cnn_type', type=str, default='STANDARD', metavar='CNN_TYPE',
                    help='cnn type (default: STANDARD)')

# train args
parser.add_argument('--train.epochs', type=int, default=30, metavar='NEPOCHS',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--train.optim_method', type=str, default='Adam', metavar='OPTIM',
                    help='optimization method (default: Adam)')
parser.add_argument('--train.learning_rate', type=float, default=0.00001, metavar='LR',
                    help='learning rate (default: 0.00001)')
parser.add_argument('--train.decay_every', type=int, default=20, metavar='LRDECAY',
                    help='number of epochs after which to decay the learning rate')
default_weight_decay = 0.0
parser.add_argument('--train.weight_decay', type=float, default=default_weight_decay, metavar='WD',
                    help="weight decay (default: {:f})".format(default_weight_decay))
parser.add_argument('--train.patience', type=int, default=1000, metavar='PATIENCE',
                    help='number of epochs to wait before validation improvement (default: 1000)')

# log args
default_fields = 'loss,acc'
parser.add_argument('--log.fields', type=str, default=default_fields, metavar='FIELDS',
                    help="fields to monitor during training (default: {:s})".format(default_fields))
default_exp_dir = 'results/new-results'
parser.add_argument('--log.exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                    help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))

parser.add_argument('--args_path', type=str, default='NULL', metavar='ARGS_PATH',
                    help="path to the argument file (default: NULL - use passed args instead)")

parser.add_argument('--train.save_all', action='store_true', help="save all models (default: False)")

args = vars(parser.parse_args())

if args['args_path'] != 'NULL':
    args_dict = json.load(open(args['args_path']))
    for name, arg_vals in args_dict.items():
        print(f"{name}: {arg_vals['description']}")
        arg_vals["log.exp_dir"] = f"{arg_vals['log.exp_dir']}/{name}"
        main(arg_vals)
else:
    main(args)
