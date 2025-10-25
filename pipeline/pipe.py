from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN ASSIGN5_2_1
    total_clock_cycles = num_batches + num_partitions - 1
    for k in range(total_clock_cycles):
        schedule = []

        for j in range(num_partitions): #See if we have a batch this clock cycle for a partition
            i = k - j
            if 0 <= i < num_batches:
                schedule.append((i,j))

        yield schedule
    # END ASSIGN5_2_1

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        
        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN ASSIGN5_2_2
        micro_size = x.size(0) // self.split_size
        batches = []
        for split in range(self.split_size):
            start_idx = split * micro_size
            end_idx = (split + 1) * micro_size if split < self.split_size - 1 else x.size(0)
            batch = x[start_idx:end_idx]
            batches.append(batch)
        
        for schedule in _clock_cycles(len(batches), len(self.partitions)):
            self.compute(batches=batches, schedule=schedule)

        # Now we have all the batches move to last device, we should concatenate and return
        last_device = self.devices[-1]
        result_batches = [batch.to(last_device) for batch in batches]
        result = torch.cat(result_batches, dim=0)
        return result
        # END ASSIGN5_2_2

    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        # BEGIN ASSIGN5_2_2
        task_info = [] #Track each task

        # Launch all workers for this schedule
        for micro_batch_idx, partition_idx in schedule:
            # Get partition module and microbatch data from the schedule
            partition_module = partitions[partition_idx]
            micro_batch_data = batches[micro_batch_idx]
            # Move data to the correct device
            target_device = devices[partition_idx]
            micro_batch_data = micro_batch_data.to(target_device, non_blocking=True)

            # Create task
            def make_compute_fn(module, data):
                def compute_fn():
                    with torch.cuda.device(data.device):
                        return module(data)
                return compute_fn
            
            task = Task(make_compute_fn(partition_module, micro_batch_data))

            # Send task to worker
            in_queue = self.in_queues[partition_idx]
            in_queue.put(task)

            # Keep track of what was sent.
            task_info.append((micro_batch_idx, partition_idx))

        # Retrieve all results form workers
        for micro_batch_idx, partition_idx in task_info:
            # Receive task output
            out_queue = self.out_queues[partition_idx]
            success, result = out_queue.get()

            # Store result back to the batches
            if success:
                task_result, batch_result = result
                batches[micro_batch_idx] = batch_result
            else:
                raise RuntimeError(f"Task failed: {result}")
        # END ASSIGN5_2_2

