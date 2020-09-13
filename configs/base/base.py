from typing import List, Dict, Any


class BaseConfig:
    @staticmethod
    def condition(key: str) -> bool:
        return not key.startswith("__")

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        class_dict = cls.__dict__.items()
        return {key: value for key, value in class_dict if cls.condition(key)}

    @classmethod
    def value_list(cls) -> List[Any]:
        return list(cls.to_dict().values())

    @classmethod
    def key_list(cls) -> List[Any]:
        return list(cls.to_dict().keys())
