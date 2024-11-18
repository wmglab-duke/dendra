import argparse

import gd_config as config

parser = argparse.ArgumentParser()
parser.add_argument("--validate", action="store_true")
parser.add_argument("--model", default=config.model)

# samples
parser.add_argument("-s", "--samples", nargs="+", default=config.samples)

# GD params
parser.add_argument("--lr", type=float, default=config.lr)
parser.add_argument("--lr-decay", type=float, default=config.lr_decay)
parser.add_argument("--n-steps", type=int, default=config.n_steps)

# fiber params
parser.add_argument("--diameter", type=float, default=config.diameter)
parser.add_argument("--nodes", type=int, default=config.nodes)
parser.add_argument("--ends_only", action="store_true", default=config.ends_only)
parser.add_argument("--n_end_nodes", type=int, default=config.n_end_nodes)
parser.add_argument("--node_check", nargs="+", default=config.node_check)
parser.add_argument("--nc", type=int, default=config.nc)

# FP precision
parser.add_argument("--fp32", action="store_true", default=config.fp32)

# parse
args = parser.parse_args()
