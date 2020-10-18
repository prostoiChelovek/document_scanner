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

    def _sort_points(self) -> None:
        self._a, self._b = sorted([self._a, self._b], key=lambda pt: pt.get_distance(Point(0, 0)))

    def get_midpoint(self) -> Point:
        return (self._a + self._b) // 2

    def get_length(self) -> float:
        return self._a.get_distance(self._b)

    def __iter__(self):
        for pt in [self._a, self._b]:
            yield pt
