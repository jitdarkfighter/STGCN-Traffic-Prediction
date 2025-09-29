MAE isn't getting lower than around ~3.2 because the dataset has 228 nodes, and I choose the first 8 nodes in it for training quickly.
What happened is, this resulted in the loss of most of the important spacial features.

Increasing the number of nodes and the input and output timesteps, can make the model even more accurate and better, but has been skipped do to GPU unavailability.

### Usage Instructions
Generate datasets containing a required number of nodes using `src/smol_dataset_maker.py` by specifying the number of datasets in the script.

Play around with `NUM_INPUT_TIMESTEPS`, `NUM_OUTPUT_TIMESTEPS` and `NUM_NODES` in the train.py. 
Note: when modifying `NUM_NODES` make use the smol dataset has been created beforehand.

The spatial, temporal, convolutional, and hidden layer sizes can be tinkered with in `src/stgcn.py` and `src/tgrn.py`. 
Additionaly the **STGCNBlocks** can be added or removed to change model depth