import argparse
import torch
import json
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='baseline', choices=['baseline', 'domain_disentangle', 'clip_disentangle'])

    parser.add_argument('--target_domain', type=str, default='cartoon', choices=['art_painting', 'cartoon', 'sketch', 'photo'])
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--max_iterations', type=int, default=5000, help='Number of training iterations.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--validate_every', type=int, default=100)

    parser.add_argument('--output_path', type=str, default='.', help='Where to create the output directory containing logs and weights.')
    parser.add_argument('--data_path', type=str, default='data/PACS', help='Locate the PACS dataset on disk.')

    parser.add_argument('--cpu', action='store_true', help='If set, the experiment will run on the CPU.')
    parser.add_argument('--weights', type=json.loads, default="[12, 0.1, 0.1, 1, 1]", help="if set, the experiment will run on different weights than the default value. Format: --weights=[weight1, weight2, weight3, alpha, clip_weight]") #w1, w2, w3, alpha, clip
    parser.add_argument('--dom_gen', action='store_true', help='If set, the experiment use domain generalization.')
    parser.add_argument('--no_descr', action='store_true', help='If set, CLIP disentanglement will use samples without description for the training.')
    parser.add_argument('--test', action='store_true', help='If set, the experiment will skip training.')

    # Additional arguments can go below this line:
    #parser.add_argument('--test', type=str, default='some default value', help='some hint that describes the effect')

    # Build options dict
    opt = vars(parser.parse_args())

    if not opt['cpu']:
        assert torch.cuda.is_available(), 'You need a CUDA capable device in order to run this experiment. See `--cpu` flag.'

    now = datetime.now()
    datetime_string = now.strftime("%Y%m%d_h%H%M%S")
    opt['output_path'] = f'{opt["output_path"]}/record/{opt["experiment"]}_{opt["target_domain"]}{"" if not opt["dom_gen"] else "_domgen"}_{datetime_string}'

    return opt