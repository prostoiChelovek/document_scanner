/**
 * @file utils.cpp
 * @author Люнгрин Андрей aka prostoichelovek <iam.prostoi.chelovek@gmail.com>
 * @date 08 Jan 2021
 * @copyright MIT License
 */

#include "utils.h"

cv::Mat pixToMat(Pix *pix) {
    int width = pixGetWidth(pix);
    int height = pixGetHeight(pix);
    int depth = pixGetDepth(pix);

    cv::Mat mat(cv::Size(width, height), depth == 1 ? CV_8UC1 : CV_8UC3);

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            if (depth == 1) {
                l_uint32 val;
                pixGetPixel(pix, x, y, &val);
                mat.at<uchar>(cv::Point(x, y)) = static_cast<uchar>(255 * val);
            } else {
                l_int32 r, g, b;
                pixGetRGBPixel(pix, x, y, &r, &g, &b);

                cv::Vec3b color(b, g, r);
                mat.at<cv::Vec3b>(cv::Point(x, y)) = color;
            }
        }
    }

    return mat;
}



std::vector<std::vector<cv::Point>> removeSmallObjects(cv::Mat &img) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<double> areas;
    std::transform(contours.begin(), contours.end(), std::back_inserter(areas),
                   [](std::vector<cv::Point> &c) -> double { return cv::contourArea(c); });

    std::vector<double> areasSorted = areas;
    std::sort(areasSorted.begin(), areasSorted.end());

    double areaThreshould = 0;
    if (areasSorted.size() >= 2) {
        areaThreshould = std::accumulate(areasSorted.end() - 3, areasSorted.end() - 1, 0) / 2;
    }

    for (int i = contours.size() - 1; i >= 0; --i) { // iterating in reverse to avoid problems with erasing
        // if area is less than threshould and contour doesn't have a parent
        if (areas[i] < areaThreshould && hierarchy[i][3] == -1) {
            cv::Rect contourRect = cv::boundingRect(contours[i]);
            cv::rectangle(img, contourRect, {0}, cv::FILLED);

            contours.erase(contours.begin() + i);
        }
    }

    return contours;
}
