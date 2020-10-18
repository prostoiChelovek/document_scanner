import math
import sys
import random
from typing import List

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
    lines = sort_lines(lines)

    res = []

    while len(lines) > 0:
        line = lines[0]

        lines.remove(line)

        matches = [line]
        for line_b in lines[:]:
            area = get_triangle_area(line.a, line.b, line_b.a)
            line_lens = [l.length for l in [line, line_b]]
            ratio = max(line_lens) / min(line_lens)
            max_area = (min(line_lens) + (max(line_lens) / (ratio / 2))) / 2
            if area <= max_area and (line.horizontal == line_b.horizontal):
                matches.append(line_b)
                lines.remove(line_b)

        res.append(matches)

    return res


def doStuff(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(gray, 170, 255)

    cnts = find_contours(edges)
    lines = get_lines(cnts)
    collinear = group_lines(lines)

    i = 0
    for lines in collinear:
        img_draw = img#.copy()

        color = tuple(random.randint(0, 255) for i in range(3))
        for line in lines:
            cv2.line(img_draw, *tuple(line), color, 2)
            cv2.putText(img_draw, str(i), tuple(line.a), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255))
        i += 1

        cv2.imshow("test", img_draw)
        # cv2.imshow("test", cv2.resize(img_draw, (img.shape[1] * 3, img.shape[0] * 3)))
        cv2.imshow("edges", edges)
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
