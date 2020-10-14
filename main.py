import math
import sys
import random

import cv2 as cv2
import numpy as np

from DataLoader import TestImagesLoader


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


def get_distance(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def get_line_length(line):
    return get_distance(line[0], line[1])


def get_line_midpoint(line):
    a, b = line
    return (int(a[0] + b[0]) // 2), int((a[1] + b[1]) // 2)


def get_line_slope(line):
    a, b = line[0], line[1]
    try:
        return (b[1] - a[1]) / (b[0] - a[0])
    except ZeroDivisionError:
        return math.inf


def get_centerline(box):
    pts = cv2.boxPoints(box)

    box_lines = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    bottom, right, top, left = box_lines

    if get_line_length(left) < get_line_length(top):
        lines = (left, right)
    else:
        lines = (top, bottom)

    return tuple([get_line_midpoint(x) for x in lines])


def get_lines(contours):
    boxes = [cv2.minAreaRect(contour) for contour in contours]

    centerlines = [get_centerline(box) for box in boxes]

    return centerlines


def sort_lines(lines):
    res = sorted(lines, key=lambda line: get_distance((0, 0), line[0]))
    return [sorted(line, key=lambda point: get_distance((0, 0), point)) for line in res]


def is_point_in_box(pt, box):
    a, b = box
    return (a[0] < pt[0] < b[0]) and (a[1] < pt[1] < b[1])


def is_line_in_box(line, box):
    return all(is_point_in_box(pt, box) for pt in line)


def get_overlap_area(box_a, box_b):
    # https://stackoverflow.com/questions/9324339/how-much-do-two-rectangles-overlap/9325084

    (ax1, ay1), (ax2, ay2) = box_a
    (bx1, by1), (bx2, by2) = box_b
    return max(0, min(bx2, ax2) - max(bx1, ax1)) * max(0, min(by2, ay2) - max(by1, ay1))


def get_box_area(box):
    (x1, y1), (x2, y2) = box
    return abs(x2 - x1) * abs(y2 - y1)


def is_line_horizontal(line):
    x_distance, y_distance = [abs(line[0][i] - line[1][i]) for i in range(2)]
    return x_distance >= y_distance


def get_wide_bound_of_line(line, box_margin, img_size):
    a, b = line

    if is_line_horizontal(line):  # horizontal
        a, b = sorted(line, key=lambda pt: pt[1])
        y1 = a[1] - box_margin
        y2 = b[1] + box_margin
        return ((0, y1), (img_size[0], y2)), True
    else:  # vertical
        a, b = sorted(line, key=lambda pt: pt[0])
        x1 = a[0] - box_margin
        x2 = b[0] + box_margin
        return ((x1, 0), (x2, img_size[1])), False


def get_triangle_area(a, b, c):
    # https://www.mathopenref.com/coordtrianglearea.html

    return abs((a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2)


def find_collinear(img, img_size, lines):
    lines = sort_lines(lines)

    box_margin = 10

    res = []
    img_draw = img.copy()

    while len(lines) > 0:
        line = lines[0]

        box, is_horizontal = get_wide_bound_of_line(line, box_margin, img_size)

        lines.remove(line)

        matches = [line]
        for line_b in lines[:]:
            area = get_triangle_area(line[0], line[1], line_b[0])
            line_lens = [get_line_length(l) for l in [line, line_b]]
            ratio = max(line_lens) / min(line_lens)
            max_area = (min(line_lens) + (max(line_lens) / (ratio / 2))) / 2
            if area <= max_area and (is_line_horizontal(line) == is_line_horizontal(line_b)):
                matches.append(line_b)
                lines.remove(line_b)
                print("matched")

            #if len(res) < 7 or not (is_line_horizontal(line) == is_line_horizontal(line_b)):
            #    continue

            print(area, max_area, ratio, max(line_lens) / (ratio / 2))

            cv2.line(img_draw, *line, (255, 0, 0), 2)
            cv2.line(img_draw, *line_b, (0, 0, 255), 2)
            triangle = [line[0], line[1], line_b[0]]
            for i in range(3):
                a, b = triangle[i], triangle[(i + 1) % 3]
                cv2.line(img_draw, a, b, (0, 255, 0))
            #cv2.imshow("triangle", img_draw)
            #cv2.waitKey(0)
            img_draw = img.copy()

        res.append(matches)

    return res


def doStuff(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(gray, 170, 255)

    cnts = find_contours(edges)
    lines = get_lines(cnts)
    collinear = find_collinear(img, gray.shape[::-1], lines)

    i = 0
    for lines in collinear:
        img_draw = img#.copy()

        color = tuple(random.randint(0, 255) for i in range(3))
        for line in lines:
            box, is_horizontal = get_wide_bound_of_line(line, 10, gray.shape[::-1])

            # cv2.rectangle(img_draw, box[0], box[1], (0, 0, 125))

            cv2.line(img_draw, line[0], line[1], color, 2)
            cv2.putText(img_draw, str(i), (line[0][0], line[0][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255))
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
            continue
            break
