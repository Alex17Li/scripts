from dataclasses import dataclass, field
from typing import Any

from typing_extensions import Never, dataclass_transform


@dataclass
@dataclass_transform()
class BaseConfig:
    """

    Base configuration class. All configuration should extend
    from this class. The code here is very simple. It just wraps every
    config class in a `@dataclass`. Nothing too fancy.

    This makes every subclass of `BaseConfig` is automatically a `dataclass`.
    """

    def __init_subclass__(cls) -> None:
        dataclass(cls)  # type: ignore


def _raise_NotImplementedError() -> Never:
    raise NotImplementedError
from kore.configs import default


not_implemented_field: Any = field(default_factory=_raise_NotImplementedError)

class A(BaseConfig):
    a: int = -1

class B(BaseConfig):
    child: A = default(A)
    a: int = -1
    def __post_init__(self) -> None:
        if self.child.a == -1:
            self.child.a = self.a
class C(BaseConfig):
    child: B = default(B)
    a: int = 4
    def __post_init__(self) -> None:
        if self.child.a == -1:
            self.child.a = self.a
c = C()
print(c.a)
print(c.child.a)
print(c.child.child.a)
