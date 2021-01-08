/**
 * @file preprocessing.cpp
 * @author Люнгрин Андрей aka prostoichelovek <iam.prostoi.chelovek@gmail.com>
 * @date 08 Jan 2021
 * @copyright MIT License
 */

#include "preprocessing.h"

void removePen(cv::Mat &img, cv::Mat const &gray) {
    cv::Mat grayBgr;
    cv::cvtColor(gray, grayBgr, cv::COLOR_GRAY2BGR);

    cv::Mat nonGray = img - grayBgr;
    cv::transform(nonGray, nonGray, cv::Matx13f::all(1)); // sum channels

    cv::Mat nonGrayMask = nonGray > 10;

    cvv::debugFilter(gray, nonGray, CVVISUAL_LOCATION, "non-gray");

    cv::Mat closingKernel = cv::getStructuringElement(cv::MORPH_RECT, {10, 10});
    cv::dilate(nonGrayMask, nonGrayMask, closingKernel);
    cv::morphologyEx(nonGrayMask, nonGrayMask, cv::MORPH_CLOSE, closingKernel, {-1, -1}, 2);

    cvv::debugFilter(nonGrayMask, nonGrayMask, CVVISUAL_LOCATION, "non-gray mask");

    auto contoursToRemove = removeSmallObjects(nonGrayMask);

    cv::Mat blackMask;
    cv::inRange(img, cv::Scalar::all(0), cv::Scalar::all(100), blackMask);
    cv::Mat blackMaskMasked = cv::Mat::zeros(blackMask.size(), blackMask.type());

    img = gray;

    for (auto const &cnt : contoursToRemove) {
        cv::Rect contourRect = cv::boundingRect(cnt);

        cv::rectangle(img, contourRect, {255}, cv::FILLED);

        cv::Mat tmp = cv::Mat::zeros(blackMaskMasked.size(), blackMaskMasked.type());
        cv::rectangle(tmp, contourRect, {255}, cv::FILLED);
        blackMaskMasked |= blackMask & tmp;
    }

    cvv::showImage(blackMaskMasked, CVVISUAL_LOCATION, "restoration mask");

    img.setTo(0, blackMask); // try to restore lines, that were accidentally hidden
}

/**
 *  make "almost white" pixels white
 */
void whiten(cv::Mat &img) {
    cv::Mat tmpThreshould;
    cv::threshold(img, tmpThreshould, 200, 255, cv::THRESH_BINARY);
    cv::erode(tmpThreshould, tmpThreshould,
              cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));
    img.setTo(255, tmpThreshould);
}

/**
 * @see https://stackoverflow.com/a/33971525/9577873
 */
void sharpen(cv::Mat &img, double sigma, double threshold, double amount) {
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(), sigma, sigma);
    cv::Mat lowContrastMask = abs(img - blurred) < threshold;
    cv::Mat sharpened = img * (1 + amount) + blurred * (-amount);
    img.copyTo(sharpened, lowContrastMask);

    img = sharpened;
}

void preprocess(cv::Mat &img) {
    cv::Mat original = img.clone();

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    removePen(img, gray);
    DebugFilter("pen removed")

    whiten(img);
    DebugFilter("whitened")

    double sigma = 11, threshold = 10, amount = 10;
    sharpen(img, sigma, threshold, amount);
    DebugFilter("sharpened")

    // cv::fastNlMeansDenoising(img, img, 10, 7, 21);
    // DebugFilter("denoised")

    cv::threshold(img, img, 100, 255, cv::THRESH_BINARY);
    DebugFilter("thresholded");
}
