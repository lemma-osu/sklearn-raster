from __future__ import annotations

from typing import TYPE_CHECKING, Generic

from ..types import RT, MaybeTuple, T

if TYPE_CHECKING:
    from collections.abc import Callable, Collection


class _UfuncResult(Generic[T]):
    """A monadic wrapper around a ufunc result as a single value or tuple of values."""

    def __init__(self, result: MaybeTuple[T]):
        self._items: tuple[T, ...] = result if isinstance(result, tuple) else (result,)
        if len(self) == 0:
            raise ValueError("UfuncResult must be non-empty.")

    def __len__(self) -> int:
        return len(self._items)

    def map(self, func: Callable[..., RT], **kwargs) -> _UfuncResult[RT]:
        """Map func over the results."""
        return _UfuncResult(tuple(func(item, **kwargs) for item in self._items))

    def zip_map(
        self,
        func: Callable[..., RT],
        *iters: Collection[object],
        **kwargs,
    ) -> _UfuncResult[RT]:
        """Zip results with additional iterables, then map the function over them."""
        if not iters:
            return self.map(func, **kwargs)

        result_length = len(self)
        iter_lengths = [len(it) for it in iters]
        if not all(result_length == il for il in iter_lengths):
            raise ValueError(
                "All iterables passed to `zip_map` must have the same length. "
                f"Result has length {result_length} but iterables have lengths "
                f"{iter_lengths}."
            )
        return _UfuncResult(
            tuple(
                func(*args, **kwargs) for args in zip(self._items, *iters, strict=True)
            )
        )

    def unwrap(self) -> MaybeTuple[T]:
        """Unwrap the result to a single value or a tuple of multiple values."""
        return self._items if len(self) > 1 else self._items[0]
