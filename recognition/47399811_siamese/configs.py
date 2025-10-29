"""
Contains all of the models hyperparameters
"""
from dataclasses import dataclass

@dataclass
class ModelConfig:
    epochs: int
    learning_rate: float

BATCH_SIZE = 32

siamese_config = ModelConfig(
    epochs=25,
    learning_rate=1e-3,
)

classifier_config = ModelConfig(
    epochs=30,
    learning_rate=1e-3,
)