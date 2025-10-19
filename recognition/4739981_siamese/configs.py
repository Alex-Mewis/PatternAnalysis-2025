"""
Contains all of the models hyperparameters
"""
from dataclasses import dataclass

@dataclass
class ModelConfig:
    epochs: int
    learning_rate: float


siamese_config = ModelConfig(
    epochs=8,
    learning_rate=1e-3,
)

classifier_config = ModelConfig(
    epochs=4,
    learning_rate=1e-3,
)