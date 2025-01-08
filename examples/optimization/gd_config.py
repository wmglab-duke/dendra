"""GD config"""

# model
model = "MRG"

# samples
samples = "all"

# GD params
lr = 2.0
lr_decay = 0.6
n_steps = 200

loss_v = "axnode_myel.m"

# models params
diameter = 5.7
nodes = 101
ends_only = True
n_end_nodes = 10
node_check = [10, 91]

nc = 6

# FP precision
fp32 = False
