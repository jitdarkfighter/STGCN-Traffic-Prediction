Paper: https://arxiv.org/pdf/1709.04875

MAE isn't getting lower than around ~2.78 because the dataset has 228 nodes, and I choose the first 8 nodes in it for training quickly.
What happened is, this resulted in the loss of most of the important spacial features.
MSE (normalized): 0.056945
MAE (original units): 2.785664
RMSE (original units): 3.870692
MAPE (%): 7.1489

Note: The paper had
MAE: 2.26
RMSE: 4.07
MAPE: 5.24

Increasing the number of nodes and the input and output timesteps, can make the model even more accurate and better, but has been skipped do to GPU unavailability.

For Graphs with increased number of nodes, Chebyshev convolution can be used. It aggregates info from neighbours upto K hops away in one node. A sweet spot would be `K=2 or K=3` after that we see diminishing returns and even worse generlization.

### Usage Instructions

**This model works best for 4 to 10 nodes. Increase the model parameters and STGCN block depth for better performance in more number of nodes.**

Generate datasets containing a required number of nodes using `src/smol_dataset_maker.py` by specifying the number of datasets in the script.

Play around with `NUM_INPUT_TIMESTEPS`, `NUM_OUTPUT_TIMESTEPS` and `NUM_NODES` in the train.py. 
Note: when modifying `NUM_NODES` make sure the smol dataset has been created beforehand.

The spatial, temporal, convolutional, and hidden layer sizes can be tinkered with in `src/stgcn.py` and `src/tgrn.py`. 
Additionally the **STGCNBlocks** can be added or removed to change model depth. **The paper used 3 blocks. This implementation uses 2.**