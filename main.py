import math
import sys

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


def get_line_length(line):
    a, b = line[0], line[1]
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def get_line_midpoint(line):
    a, b = line[0], line[1]
    return (int(a[0] + b[0]) // 2), int((a[1] + b[1]) // 2)


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


def doStuff(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                  cv2.THRESH_BINARY, 15, 2)

    edges = cv2.Canny(gray, 170, 255)

    cnts = find_contours(edges)
    lines = get_lines(cnts)

    """
    for c in cnts:
        box = cv2.minAreaRect(c)
        pts = cv2.boxPoints(box)
        for i in range(len(pts)):
            b = (i + 1) % len(pts)
            cv2.line(img, tuple(pts[i]), tuple(pts[b]), (0, 0, 255))
    """

    for pt1, pt2 in lines:
        cv2.line(img, pt1, pt2, (255, 0, 255))

    cv2.imshow("test", img)
    cv2.imshow("edges", edges)


loader = TestImagesLoader("data/0", rotate=cv2.ROTATE_90_COUNTERCLOCKWISE)
for img in loader:
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    doStuff(img)

    while True:
        k = cv2.waitKey(0)
        if k == 27:
            sys.exit(0)
        elif k == 32:
            break
