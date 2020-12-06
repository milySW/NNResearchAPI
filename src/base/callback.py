from typing import List

from pytorch_lightning.callbacks.base import Callback


class BaseCallback(Callback):
    @staticmethod
    def check_variant(variants: List[str]):
        supported = ["val", "test", ""]
        info = f"Variants not supported. Supported variants: {supported}"

        con = [True if variant in supported else False for variant in variants]
        assert all(con), info

        return supported
