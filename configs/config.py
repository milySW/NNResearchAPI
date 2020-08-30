from configs.config_template import DefaultConfig
from configs.training_template import DefaultTraining
from configs.models.resnet_template import DefaultResnet


config = DefaultConfig(model=DefaultResnet(), training=DefaultTraining())
