from configs import (  # noqa
    DefaultAugmentations,
    DefaultBindedHooks,
    DefaultCallbacks,
    DefaultConfig,
    DefaultEvaluations,
    DefaultMetrics,
    DefaultOptimizersAndSchedulers,
    DefaultPostprocessors,
    DefaultPrediction,
    DefaultPreprocessors,
    DefaultResnet,
    DefaultRNN,
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
    evaluations=DefaultEvaluations,
)
