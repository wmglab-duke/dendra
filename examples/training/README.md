## :rocket: Training a model

In the provided example, `axonml` trains approximations to the MRG[^2] myelinated fiber model. Check `axonml/models/README.md` for instructions on how to implement approximations of other fiber models.

Training configurations can be modified by changing the relevant values in `config.py`:

|Variable|Description|
|---|---|
|`model`|Surrogate fiber model `class` to train. **Default: `axonml.models.SMF`**.|
|`cuda`|Whether to use GPU. **Default: True if GPU is available, False otherwise**.|
|`fp32`|Whether to use single-precision floating point arithmetic. If not, double precision is used. **Default: False**.|
|`nodes`|Nodes of Ranvier per axon. **Default: 53**.|
|`dt`|Simulation timestep [ms]. **Default: 0.005**.|
|`train_dset`|Path to training dataset.|
|`valid_dset`|Path to validation dataset.|
|`m_states`|The state variable names from the model that will be used to compute the loss. Must agree with the order of `states`. **Default: ['axnode_myel.m', 'axnode_myel.h', 'axnode_myel.p', 'axnode_myel.s', 'v']**.|
|`states`|The state variables in the training / validations dataset(s) and the order in which they will be concatenated. Must agree with the order of `m_states`. **Default: ['m', 'h', 'p', 's', 'v']**.|
|`epochs`|Number of training epochs. **Default: 5**.|
|`truncation_length`|Sequence length over which to perform truncated backpropagation through time. **Default: 50**.|
|`lr`|Adam optimizer learning rate. **Default: $3\times10^{-5}$**.|
|`grad_accumulation`|Whether to use gradient accumulation over disjoint chunks in truncated backpropagation through time. **Default: False**.|
|`to_train`|Parameters to train.|
|`to_train_groups`|Parameter groups to train.|
|`train_n_idx`|Total # training set batches to use per training epoch (see `./axonml/data/generate_data.py` - `n_batches`). **Default: 64**.|
|`val_n_idx`| # validation set batches to use per round of validation. **Default: 8**.|
|`train_chunk_size`| # training set batches to use per gradient-descent step. (Should be a factor of `train_n_idx`). **Default: 2**.|
|`val_chunk_size`| # validation set batches to use per validation step. **Default: 8**.|
|`sampling`|Whether to downsample training set in time, and by how much (sample every `sampling` timesteps). **Default: None**.|
|`postfix`|List of model parameters to display in progressbar as training progresses. **Default: None**.|
|`save_every`|Save model parameters every `save_every` minibatch iterations. **Default: 32**.|
|`save_dir`|Location into which to save model parameters. **Default: `./checkpoints/`**.|

Once you've set variables appropriately in `config.py`, you can initiate training:

```bash
(base) foo@bar : ~ $ cd /path/to/cloned/repository/examples/training
(base) foo@bar : axonml/examples/training $ conda activate axonml
(axonml) foo@bar : axonml/examples/training $ python train.py
```
