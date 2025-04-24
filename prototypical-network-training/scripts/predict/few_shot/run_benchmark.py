import argparse
import os
from benchmark_throughput import main

parser = argparse.ArgumentParser(description='Benchmark throughput for few-shot prototypical networks')
default_dataset = os.path.join(os.path.dirname(__file__), '../../../datasets/combined_ds')
default_model_path = 'results/std-cnn/best_model.pt'
parser.add_argument('--model.model_path', type=str, default=default_model_path, metavar='MODELPATH',
                    help=f"location of pretrained model (default: {default_model_path})")
model_options_path = 'results/std-cnn/opt.json'
parser.add_argument('--model.model_options_path', type=str, default=model_options_path, metavar='MODELOPTIONS',
                    help=f"location of model options json (default: {model_options_path})")
parser.add_argument('--data.dataset', type=str, default=default_dataset, metavar='DS',
                    help="override dataset name (default: combined_ds)")
parser.add_argument('--data.test_shot', type=int, default=None, help='override number of support shots per class')
parser.add_argument('--data.test_query', type=int, default=None, help='override number of query examples per class')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--mps', action='store_true', help='use Apple MPS device')

args = parser.parse_args()
config = vars(args)
config['data.cuda'] = config.pop('cuda')
config['data.mps'] = config.pop('mps')

main(config)
