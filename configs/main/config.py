from configs import (
    DefaultBindedHooks,
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
    hooks=DefaultBindedHooks,
    prediction=DefaultPrediction,
    evaluations=DefaultEvaluation,
)
