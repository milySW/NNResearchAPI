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
    DefaultTransformations,
)

config = DefaultConfig(
    model=DefaultResnet,
    training=DefaultTraining,
    optimizers=DefaultOptimizersAndSchedulers,
    metrics=DefaultMetrics,
    callbacks=DefaultCallbacks,
    hooks=DefaultBindedHooks,
    transformations=DefaultTransformations,
    prediction=DefaultPrediction,
    evaluations=DefaultEvaluation,
)
