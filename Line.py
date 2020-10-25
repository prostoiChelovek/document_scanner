from __future__ import annotations

from Point import Point


class Line:
    def __init__(self, a: Point, b: Point):
        self._a = a
        self._b = b

        self._sort_points()

    @property
    def a(self) -> Point:
        return self._a

    @property
    def b(self) -> Point:
        return self._b

    @a.setter
    def a(self, value: Point) -> None:
        self._a = value

        self._sort_points()

    @b.setter
    def b(self, value: Point) -> None:
        self._b = value

        self._sort_points()

    @property
    def horizontal(self) -> bool:
        distance = self._a - self._b
        return distance.x <= distance.y

    @property
    def length(self) -> float:
        return self._a.get_distance(self._b)

    @property
    def midpoint(self) -> Point:
        return (self._a + self._b) // 2

    def intersection(self, line2: Line) -> Point:
        # https://stackoverflow.com/a/20677983/9577873

        xdiff = (self.a.x - self.b.x, line2.a.x - line2.b.x)
        ydiff = (self.a.y - self.b.y, line2.a.y - line2.b.y)

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return Point(-1, -1)

        d = (det(*self), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return Point(x, y)

    def _sort_points(self) -> None:
        self._a, self._b = sorted([self._a, self._b], key=lambda pt: pt.get_distance(Point(0, 0)))

    def __iter__(self):
        for pt in [self._a, self._b]:
            yield tuple(pt)

    def __str__(self):
        return str(tuple(self))

    def __repr__(self):
        return str(self)
