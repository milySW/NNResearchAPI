from configs import (
    DefaultAugmentations,
    DefaultBindedHooks,
    DefaultCallbacks,
    DefaultConfig,
    DefaultEvaluation,
    DefaultMetrics,
    DefaultOptimizersAndSchedulers,
    DefaultPostprocessors,
    DefaultPrediction,
    DefaultPreprocessors,
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
    preprocessors=DefaultPreprocessors,
    augmentations=DefaultAugmentations,
    postprocessors=DefaultPostprocessors,
    prediction=DefaultPrediction,
    evaluations=DefaultEvaluation,
)
