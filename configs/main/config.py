from configs import (
    DefaultCallbacks,
    DefaultConfig,
    DefaultEvaluation,
    DefaultMetrics,
    DefaultOptimizersAndSchedulers,
    DefaultPrediction,
    DefaultResnet,
    DefaultTraining,
)

config = DefaultConfig(
    model=DefaultResnet,
    training=DefaultTraining,
    optimizers=DefaultOptimizersAndSchedulers,
    metrics=DefaultMetrics,
    callbacks=DefaultCallbacks,
    prediction=DefaultPrediction,
    evaluations=DefaultEvaluation,
)
