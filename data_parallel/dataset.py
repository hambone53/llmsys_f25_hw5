from random import Random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


class Partition():
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        '''Given index, get the data according to the partitioned index'''
        # BEGIN ASSIGN5_1_1
        return self.data[self.index[index]]
        # END ASSIGN5_1_1

class DataPartitioner():
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        ''' Create indices for different partitions
        1. Create indices and use `rng` to shuffle indices
        2. Create different partitions of indices according to `sizes` and store in `self.partitions`
        '''
        # BEGIN ASSIGN5_1_1
        indices = list(range(len(self.data)))
        rng.shuffle(indices)
        start_idx = 0
        for size in sizes:
            partition_size = int(size * len(indices))
            partition_indices = indices[start_idx:start_idx + partition_size]
            self.partitions.append(Partition(data, partition_indices))
            start_idx += partition_size
        # END ASSIGN5_1_1

    def use(self, partition):
        ''' Return a simple dataset class `Partiton` by original data and partitioned indices

        Just one line of code. Think it simply.
        '''
        # BEGIN ASSIGN5_1_1
        return self.partitions[partition]
        # END ASSIGN5_1_1

def partition_dataset(rank, world_size, dataset, batch_size=128, collate_fn=None):
    """ Partitioning training dataset of the Machine Translation

    Returns:
        DataLoader: partitioned dataloader
    
    Hint:
    1. Calculate the partitioned batch size
    2. Create a partitioner class `DataPartitioner` with dataset and the list of partitioned sizes
    3. Get the current partition dataset given `rank`, use the `use` function in DataPartitioner
    4. Wrap the dataset with `DataLoader`, remember to customize the `collate_fn`
    """
    # BEGIN ASSIGN5_1
    partition_batch_size = batch_size // world_size
    sizes = [1.0 / world_size] * world_size
    data_partitioner = DataPartitioner(dataset, sizes)
    parition_dataset = data_partitioner.use(rank)
    dataloader = DataLoader(dataset=parition_dataset, batch_size=partition_batch_size, collate_fn=collate_fn)
    return dataloader
    # END ASSIGN5_1
