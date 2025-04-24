import json
import math
import sys

import torch
import torchnet as tnt

from protonets.utils import filter_opt
import protonets.utils.data as data_utils
import protonets.utils.model as model_utils

def main(opt, verbose=True):
    model = torch.load(opt['model.model_path'], weights_only=False)
    model.eval()
    model_opt_file = opt['model.model_options_path']
    with open(model_opt_file, 'r') as f:
        model_opt = json.load(f)
    
    model_opt['model.x_dim'] = map(int, model_opt['model.x_dim'].split(','))
    model_opt['log.fields'] = model_opt['log.fields'].split(',')
    data_opt = { 'data.' + k: v for k,v in filter_opt(model_opt, 'data').items() }
    if 'data.dataset' in opt:
        data_opt['data.dataset'] = opt['data.dataset']

    episode_fields = {
        'data.test_way': 'data.way',
        'data.test_shot': 'data.shot',
        'data.test_query': 'data.query',
        'data.test_episodes': 'data.train_episodes'
    }

    for k,v in episode_fields.items():
        if opt[k] != 0:
            data_opt[k] = opt[k]
        elif model_opt[k] != 0:
            data_opt[k] = model_opt[k]
        else:
            data_opt[k] = model_opt[v]

    if verbose:
        print("Evaluating {:d}-way, {:d}-shot with {:d} query examples/class over {:d} episodes".format(
            data_opt['data.test_way'], data_opt['data.test_shot'],
            data_opt['data.test_query'], data_opt['data.test_episodes']))

    torch.manual_seed(1234)
    if data_opt['data.cuda']:
        torch.cuda.manual_seed(1234)
    elif data_opt.get('data.mps', False):
        torch.mps.manual_seed(1234)

    data = data_utils.load(data_opt, ['test'])

    if data_opt['data.cuda']:
        model.cuda()
    elif data_opt.get('data.mps', False):
        model.to('mps')
        
    meters = { field: tnt.meter.AverageValueMeter() for field in model_opt['log.fields'] }

    model_utils.evaluate(model, data['test'], meters, desc="test")
    results = {}
    
    for field,meter in meters.items():
        mean, std = meter.value()
        results[field] = mean
        if verbose: 
            print("test {:s}: {:0.6f} +/- {:0.6f}".format(field, mean, 1.96 * std / math.sqrt(data_opt['data.test_episodes'])))

    if opt.get('f1', False):
        y_true_all = []
        y_pred_all = []
        for sample in data['test']:
            _, output = model.loss(sample)
            y_true_all.extend(output['y_true'])
            y_pred_all.extend(output['y_pred'])
        labels = sorted(set(y_true_all))
        f1_scores = []
        for label in labels:
            TP = sum(1 for yt, yp in zip(y_true_all, y_pred_all) if yt == label and yp == label)
            FP = sum(1 for yt, yp in zip(y_true_all, y_pred_all) if yt != label and yp == label)
            FN = sum(1 for yt, yp in zip(y_true_all, y_pred_all) if yt == label and yp != label)
            precision = TP / (TP + FP) if TP + FP > 0 else 0.0
            recall = TP / (TP + FN) if TP + FN > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
            f1_scores.append(f1)
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        results['f1'] = macro_f1
        if verbose:
            print("F1 score: {:0.6f}".format(macro_f1))

    return results
