from enum import unique, Enum


@unique
class SPLIT_DATA_PROTOCOL(Enum):
    TRAIN_I_TEST_II = "TRAIN_I_TEST_II"
    TRAIN_II_TEST_I = "TRAIN_II_TEST_I"
    TRAIN_ALL_TEST_ALL = "TRAIN_ALL_TEST_ALL"

    def __str__(self):
        return self.value


@unique
class LOAD_TASK_MODE(Enum):
    LOAD = "LOAD"
    NO_LOAD = "NO_LOAD"
    def __str__(self):
        return self.value

@unique
class DISTILLATION_TYPE(Enum):
    SOFTMAX = "SOFTMAX"
    PAIR_LOSS = "PAIR_LOSS"

@unique
class PRIOR_MODE(Enum):
    PGD_GRAD = "PGD_GRAD"
    SIMULATE_GRAD = "SIMULATE_GRAD"