import time
import json
import argparse
import torch
from protonets.utils import filter_opt
import protonets.utils.data as data_utils

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None

def main(config):
    model = torch.load(config['model.model_path'], weights_only=False)
    model.eval()
    if config.get('data.cuda'):
        device = torch.device('cuda')
    elif config.get('data.mps'):
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model.to(device)

    with open(config['model.model_options_path'], 'r', encoding='utf-8') as f:
        model_opt = json.load(f)
    model_opt['model.x_dim'] = list(map(int, model_opt['model.x_dim'].split(',')))
    model_opt['log.fields'] = model_opt['log.fields'].split(',')

    data_opt = {'data.' + k: v for k, v in filter_opt(model_opt, 'data').items()}
    if 'data.dataset' in config and config['data.dataset']:
        data_opt['data.dataset'] = config['data.dataset']
        if config.get('data.test_shot') is not None:
            data_opt['data.test_shot'] = config['data.test_shot']
        if config.get('data.test_query') is not None:
            data_opt['data.test_query'] = config['data.test_query']

    data = data_utils.load(data_opt, ['test'])
    loader = data['test']

    if get_model_complexity_info is not None:
        sample0 = next(iter(loader))
        img = sample0['xs'][0][0].unsqueeze(0).to(device)
        macs, params = get_model_complexity_info(
            model.encoder, (img.size(1), img.size(2), img.size(3)),
            as_strings=False, print_per_layer_stat=False, verbose=False
        )
        print(f"Model Parameters: {params:,}")
        print(f"Model MACs: {macs:,}")
        print(f"Model FLOPs: {macs*2:,}")
    else:
        print("ptflops package not installed; skipping FLOPS calculation.")

    with torch.no_grad():
        for i, sample in enumerate(loader):
            xs, xq = sample['xs'].to(device), sample['xq'].to(device)
            _ = model.loss({'xs': xs, 'xq': xq})
            if i >= 2:
                break
    total_images = 0
    start = time.time()
    with torch.no_grad():
        for sample in loader:
            xs, xq = sample['xs'].to(device), sample['xq'].to(device)
            batch_images = xs.size(0) * (xs.size(1) + xq.size(1))
            _ = model.loss({'xs': xs, 'xq': xq})
            total_images += batch_images
    elapsed = time.time() - start

    print(f"Processed {total_images} images in {elapsed:.4f} seconds")
    print(f"Throughput: {total_images/elapsed:.2f} images/sec")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark model throughput')
    parser.add_argument('--model.model_path', required=True, help='path to saved model')
    parser.add_argument('--model.model_options_path', required=True,
                        help='path to model options json')
    parser.add_argument('--data.dataset', default=None, help='override dataset name')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--mps', action='store_true', help='use MPS')
    args = parser.parse_args()
    cfg = vars(args)
    cfg['data.cuda'] = cfg.pop('cuda')
    cfg['data.mps'] = cfg.pop('mps')
    main(cfg)
