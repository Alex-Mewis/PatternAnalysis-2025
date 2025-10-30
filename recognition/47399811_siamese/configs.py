"""
Contains all of the models hyperparameters
"""
from dataclasses import dataclass

@dataclass
class ModelConfig:
    epochs: int
    learning_rate: float

BATCH_SIZE = 32

config = ModelConfig(
    epochs=25,
    learning_rate=0.01,
)
