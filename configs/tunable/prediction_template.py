import configs


class DefaultPrediction(configs.BaseConfig):
    """
    Config responsible for setting prediction parameters.

    Parameters:

        int batch_size: Number of elements in one batch
    """

    batch_size: int = 500
