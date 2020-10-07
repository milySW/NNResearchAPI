from configs import (
    DefaultCallbacks,
    DefaultConfig,
    DefaultEvaluations,
    DefaultMetrics,
    DefaultOptimizersAndSchedulers,
    DefaultResnet,
    DefaultTraining,
)

config = DefaultConfig(
    model=DefaultResnet,
    training=DefaultTraining,
    optimizers=DefaultOptimizersAndSchedulers,
    metrics=DefaultMetrics,
    callbacks=DefaultCallbacks,
    evaluations=DefaultEvaluations,
)
