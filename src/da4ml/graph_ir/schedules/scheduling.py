from dataclasses import dataclass
from typing import Callable, Type

import keras

from hgq.layers import QDense, QSum
from da4ml.cmvm import CombLogic 

from ..hardware_types import QSumLogic
from .qdense import dense_schedule, input_dense_schedule_requirement, minimum_output_shape_for_dense, minimum_input_shape_for_dense, dense_rebuilder
from .qsum import input_qsum_requirement, minimum_output_shape_for_qsum, minimum_input_shape_for_qsum, qsum_schedule, qsum_rebuilder

@dataclass
class DataSchedule():
    shape_req: Callable[[tuple], bool]
    minimum_output_shape: Callable[[tuple], tuple]
    minimum_input_shape: Callable[[tuple], tuple]
    schedule : Callable 
    rebuilder:  Callable
    buffer_type: str 
    hardware_type: Type
    
class DataScheduler():
    def __init__(self, data_schedule) -> None:
        self.data_schedule = data_schedule
            
    def call(self, x : keras.KerasTensor) -> list[keras.KerasTensor]:
        output = None
        assert self.data_schedule.shape_req(x.shape), f"Input shape {x.shape} does not meet schedule requirements"
        output = self.data_schedule.schedule(x)        
        return output

    def rebuild_tensor(self, chunks, original):
        return self.data_schedule.rebuilder(chunks, original)
    
    def __call__(self, x):
        return self.call(x)


_SCHEDULE_REGISTRY : dict[type, DataSchedule] = {
    # keras.layers.Dense : DataSchedule(shape_req=input_dense_schedule_requirement, minimum_output_shape=minimum_output_shape_for_dense, minimum_input_shape=minimum_input_shape_for_dense, schedule=dense_schedule, rebuilder=dense_rebuilder),
    QDense : DataSchedule(shape_req=input_dense_schedule_requirement, minimum_output_shape=minimum_output_shape_for_dense, minimum_input_shape=minimum_input_shape_for_dense, schedule=dense_schedule, rebuilder=dense_rebuilder, buffer_type="fifo", hardware_type=CombLogic),
    
    QSum : DataSchedule(shape_req=input_qsum_requirement, minimum_output_shape=minimum_output_shape_for_qsum, minimum_input_shape=minimum_input_shape_for_qsum, schedule=qsum_schedule, rebuilder=qsum_rebuilder, buffer_type="fifo", hardware_type=QSumLogic)
}

