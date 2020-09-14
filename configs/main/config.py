from configs import (
    DefaultConfig,
    DefaultTraining,
    DefaultResnet,
    DefaultOptimizersAndSchedulers,
    DefaultMetrics,
    DefaultCallbacks,
)


config = DefaultConfig(
    model=DefaultResnet,
    training=DefaultTraining,
    optimizers=DefaultOptimizersAndSchedulers,
    metrics=DefaultMetrics,
    callbacks=DefaultCallbacks,
)
