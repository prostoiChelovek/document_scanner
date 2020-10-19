import math
import sys
import random
from typing import List, Tuple, Optional

import cv2 as cv2
import numpy as np

from DataLoader import TestImagesLoader
from Point import Point
from Line import Line


def _find_contours_with_kernel(img, kernel):
    morphed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts[0] if len(cnts) == 2 else cnts[1]


def find_contours(img):
    # https://stackoverflow.com/a/57531326/9577873

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 2))
    horizontal = _find_contours_with_kernel(close, horizontal_kernel)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 45))
    vertical = _find_contours_with_kernel(close, vertical_kernel)

    return horizontal + vertical


def get_centerline(box) -> Line:
    pts = cv2.boxPoints(box)

    box_lines = [Line(Point.from_list(pts[i]), Point.from_list(pts[(i + 1) % len(pts)]))
                 for i in range(len(pts))]
    bottom, right, top, left = box_lines

    if left.length < top.length:
        lines = (left, right)
    else:
        lines = (top, bottom)

    return Line(*(x.midpoint for x in lines))


def get_lines(contours) -> List[Line]:
    boxes = [cv2.minAreaRect(contour) for contour in contours]

    return [get_centerline(box) for box in boxes]


def sort_lines(lines: List[Line]) -> List[Line]:
    return sorted(lines, key=lambda line: line.a.get_distance(Point(0, 0)))


def get_triangle_area(a: Point, b: Point, c: Point) -> float:
    # https://www.mathopenref.com/coordtrianglearea.html

    return abs((a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) / 2)


def group_lines(lines: List[Line]) -> List[List[Line]]:
    res = []

    while len(lines) > 0:
        line = lines[0]

        lines.remove(line)

        matches = [line]
        for line_b in lines[:]:
            # TODO: maybe it'll be better to use angles instead
            # TODO: fix fails on some close vertical lines

            area = get_triangle_area(line.a, line.b, line_b.a)
            line_lens = [l.length for l in [line, line_b]]
            ratio = max(line_lens) / min(line_lens)
            max_area = (min(line_lens) + (max(line_lens) / (ratio / 2))) / 2
            max_distance = max(line_lens) / 2
            if area <= max_area and (line.horizontal == line_b.horizontal) \
                    and line.b.get_distance(line_b.a) < max_distance:
                matches.append(line_b)
                lines.remove(line_b)

        res.append(matches)

    return res


def get_group_guide(lines: List[Line]) -> Line:
    points = []
    for line in lines:
        for pt in line:
            points.append(np.int32(np.int32(tuple(pt))))

    bounding_rect = cv2.minAreaRect(np.array(points))
    return get_centerline(bounding_rect)


def get_guides(lines: List[List[Line]]) -> List[Line]:
    return [get_group_guide(group) for group in lines]


def get_point_line_distance(line: Line, point: Point) -> float:
    # http://paulbourke.net/geometry/pointlineplane/

    u = sum((point - line.a) * (line.b - line.a)) / (np.linalg.norm(tuple(line.b - line.a)) ** 2)
    pt = line.a + (line.b - line.a) * u
    return pt.get_distance(point)


def find_table_outline(img, lines: List[Line]) -> Optional[Tuple[Line, Line, Line, Line]]:
    distance_weight = 2
    length_weight = 1
    lower_to_right_weight = 30

    pair_score_fn = lambda line_a, line_b: ((get_point_line_distance(line_a, line_b.a) * distance_weight)
                                            + (line_b.length * length_weight)) \
                                           if line_a.horizontal == line_b.horizontal else -1

    outline: List[Line] = []
    for horizontal in [False, True]:
        lines = sorted(lines, key=lambda l: l.a.y if horizontal else l.a.x)

        best_candidate: Optional[Tuple[Line, Line]] = None
        for i, line_a in enumerate(lines[:-1]):
            if line_a.horizontal != horizontal:
                continue

            pair = line_a, max(lines[i + 1:],
                               key=lambda line_b: pair_score_fn(line_a, line_b) - (line_b.a.get_distance(outline[0].b) * lower_to_right_weight if horizontal else 0)) \
                if i != len(lines) - 1 else lines[i+1]
            print(pair_score_fn(line_a, pair[1]) - (pair[1].a.get_distance(outline[0].b) * lower_to_right_weight if horizontal else 0),
                  pair[1].a.get_distance(outline[0].b) * lower_to_right_weight if horizontal else 0
                  )

            if best_candidate is None:
                best_candidate = pair
            elif line_a.horizontal == pair[1].horizontal:
                img_draw = img.copy()
                cv2.line(img_draw, *tuple(best_candidate[0]), (255, 0, 0))
                cv2.line(img_draw, *tuple(best_candidate[1]), (0, 255, 0))
                cv2.line(img_draw, tuple(best_candidate[0].midpoint), tuple(best_candidate[1].midpoint), (0, 255, 0), 2)

                best_candidate = max([best_candidate, pair],
                                     key=lambda c: pair_score_fn(*c) + c[0].length * length_weight)

                cv2.line(img_draw, *tuple(pair[0]), (0, 0, 255), 2)
                cv2.line(img_draw, *tuple(pair[1]), (255, 0, 255), 2)
                cv2.line(img_draw, tuple(pair[0].midpoint), tuple(pair[1].midpoint), (0, 0, 255), 2)
                if horizontal:
                    cv2.line(img_draw, *tuple(outline[0]), (125, 125, 255), 2)
                    cv2.circle(img_draw, tuple(pair[1].a), 5, (125, 125, 255), cv2.FILLED)

                cv2.imshow("candidates", img_draw)
                #while cv2.waitKey(0) != 32: continue

        if best_candidate is not None:
            outline += list(best_candidate)
        else:
            return None

    return tuple(outline)


def doStuff(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(gray, 170, 255)

    cnts = find_contours(edges)
    lines = get_lines(cnts)
    lines = sort_lines(lines)
    grouped = group_lines(lines)
    guides = get_guides(grouped)
    table_outline = find_table_outline(img, guides)

    for guide in table_outline:
        img_draw = img#.copy()

        color = tuple(random.randint(0, 255) for i in range(3))
        cv2.line(img_draw, *tuple(guide), color, 3)

        cv2.imshow("test", img_draw)
        # cv2.imshow("test", cv2.resize(img_draw, (img.shape[1] * 3, img.shape[0] * 3)))
        # cv2.imshow("edges", edges)
        # while cv2.waitKey(0) != 32: continue


loader = TestImagesLoader("data/0", rotate=cv2.ROTATE_90_COUNTERCLOCKWISE)
for img in loader:
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    doStuff(img)

    while True:
        k = cv2.waitKey(0)
        if k == 27:
            sys.exit(0)
        elif k == 32:
            # continue
            break
