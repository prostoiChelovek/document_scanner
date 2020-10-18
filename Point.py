from __future__ import annotations

import math
from typing import Callable, Union


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get_distance(self, b: Point) -> float:
        return math.sqrt(sum((b - self) ** 2))

    def _compare(self, other: Union[Point, int], op: Callable[[int, int], bool],
                 logic_op: Callable[[bool, bool], bool] = None) -> bool:
        if logic_op is None:
            logic_op = lambda a, b: a and b
        other_pt = other if isinstance(other, Point) else Point(other, other)

        return logic_op(op(self.x, other_pt.x), op(self.y, other_pt.y))

    def _perform_op(self, other: Union[Point, int], op: Callable[[int, int], int]) -> Point:
        other_pt = other if isinstance(other, Point) else Point(other, other)

        return Point(op(self.x, other_pt.x), op(self.y, other_pt.y))

    @classmethod
    def from_list(cls, arr):
        assert len(arr) == 2, "List should have two elements (x, y)"

        return Point(*arr)

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b)

    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b)

    def __le__(self, other):
        return self._compare(other, lambda a, b: a <= b)

    def __ge__(self, other):
        return self._compare(other, lambda a, b: a >= b)

    def __eq__(self, other):
        return self._compare(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._compare(other, lambda a, b: a != b, lambda a, b: a or b)

    def __add__(self, other):
        return self._perform_op(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._perform_op(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._perform_op(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self._perform_op(other, lambda a, b: a / b)

    def __floordiv__(self, other):
        return self._perform_op(other, lambda a, b: a // b)

    def __pow__(self, power):
        return self._perform_op(power, lambda a, b: a ** b)

    def __iter__(self):
        for val in [self.x, self.y]:
            yield int(val)

    def __str__(self):
        return str(tuple(self))

    def __repr__(self):
        return str(self)
