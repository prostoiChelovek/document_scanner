/**
 * @file utils.h
 * @author Люнгрин Андрей aka prostoichelovek <iam.prostoi.chelovek@gmail.com>
 * @date 08 Jan 2021
 * @copyright MIT License
 */

#ifndef DOCUMENT_SCANNER_UTILS_H
#define DOCUMENT_SCANNER_UTILS_H

#include <numeric>

#include <opencv2/opencv.hpp>

#include <leptonica/allheaders.h>

#ifdef CVVISUAL_DEBUGMODE
/**
 * Helper macro, which calls CVV's debugFilter, and clones `img` to `original`
 */
#define DebugFilter(name) cvv::debugFilter(original, img, CVVISUAL_LOCATION, name); \
                          original = img.clone();
#else
#define DebugFilter(name)
#endif

cv::Mat pixToMat(Pix *pix);

std::vector<std::vector<cv::Point>> removeSmallObjects(cv::Mat &img);

#endif //DOCUMENT_SCANNER_UTILS_H
