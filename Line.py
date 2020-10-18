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
        return distance.x >= distance.y

    @property
    def length(self) -> float:
        return self._a.get_distance(self._b)

    @property
    def midpoint(self) -> Point:
        return (self._a + self._b) // 2

    def _sort_points(self) -> None:
        self._a, self._b = sorted([self._a, self._b], key=lambda pt: pt.get_distance(Point(0, 0)))

    def __iter__(self):
        for pt in [self._a, self._b]:
            yield tuple(pt)
