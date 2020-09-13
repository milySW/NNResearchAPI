from configs import (
    DefaultConfig,
    DefaultTraining,
    DefaultResnet,
    DefaultOptimizers,
    DefaultMetrics,
    DefaultCallbacks,
)


config = DefaultConfig(
    model=DefaultResnet,
    training=DefaultTraining,
    optimizers=DefaultOptimizers,
    metrics=DefaultMetrics,
    callbacks=DefaultCallbacks,
)
