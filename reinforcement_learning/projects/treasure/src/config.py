from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """
    Config for treasure hunting training
    """
    size:       int   = 3
    seed:       int   = 1337

    train_episodes:      int   = 33333
    learning_rate:   float = 0.01
    discount_factor: float = 0.95

    epsilon:         float = 0.1

    test_episodes: int = 6

    def __post_init__(self):
        if self.size < 2:
            raise ValueError("World size must be >= 2")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("Learning rate must be in (0, 1]")