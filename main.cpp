#include <cstdio>
#include <iostream>
#include <numeric>

#include <opencv2/opencv.hpp>

#define CVVISUAL_DEBUGMODE
#include <opencv2/cvv.hpp>

#include <tesseract/baseapi.h>
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

struct Word {
    cv::Rect rect;
    std::string text;
    float confidence;
};

struct Line {
    cv::Point a, b;
};

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

void drawWords(cv::Mat &img, std::vector<Word> const &words, std::vector<Line> const &lines) {
    for (auto const &word : words) {
        cv::rectangle(img, word.rect, {255, 0, 0}, 2);
    }

    for (auto const &line : lines) {
        cv::line(img, line.a, line.b, {0, 0, 255}, 2);
    }

    cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);

    for (auto const &word : words) {
        cv::Point textOrig = {word.rect.y, img.size().height - word.rect.x - word.rect.width / 2 + 5};
        cv::putText(img, word.text, textOrig, cv::FONT_HERSHEY_COMPLEX, 1.4,
                    {0, 255, 0}, 2);
    }
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

int main(int argc, char *argv[]) {
    auto *api = new tesseract::TessBaseAPI();
    // https://github.com/tesseract-ocr/tessdata_best
    if (api->Init("/usr/share/tesseract-ocr/4.00/tessdata/best", "rus+eng")) {
        api->End();

        std::cerr << "Cannot initialize tesseract" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<Word> words;
    std::vector<Line> lines;

    api->SetPageSegMode(tesseract::PSM_AUTO_OSD);

    for (auto const &var : std::vector<std::string>
            {
                    "textord_tabfind_find_tables",
                    "textord_tablefind_recognize_tables"
            }) {
        api->SetVariable(var.c_str(), "true");
    }

    cv::Mat img = cv::imread("../data/0/01.png");
    cv::resize(img, img, img.size() * 2);
    cv::Mat original = img.clone();

    preprocess(img);

    api->SetImage((uchar *) img.data, img.size().width, img.size().height, img.channels(), img.step1());
    api->SetSourceResolution(300);

    std::cout << "Performing OCR..." << std::endl;
    api->Recognize(nullptr);

    tesseract::ResultIterator *ri = api->GetIterator();
    tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
    if (ri != nullptr) {
        do {
            const char *text = ri->GetUTF8Text(level);
            float conf = ri->Confidence(level);
            int x1, y1, x2, y2;
            ri->BoundingBox(level, &x1, &y1, &x2, &y2);

            cv::Rect rect(cv::Point(x1, y1), cv::Point(x2, y2));

            PolyBlockType blockType = ri->BlockType();
            if (blockType == PolyBlockType::PT_HORZ_LINE) {
                int rectXCenter = rect.x + rect.width / 2;
                cv::Point a(rectXCenter, rect.y);
                cv::Point b(rectXCenter, rect.y + rect.height);
                lines.emplace_back(Line{a, b});
            } else if (blockType == PolyBlockType::PT_VERT_LINE) {
                cv::Point a(rect.x, rect.y + rect.height / 2);
                cv::Point b(rect.x + rect.width, rect.y + rect.height / 2);
                lines.emplace_back(Line{a, b});
            } else if (PTIsTextType(blockType)) {
                words.emplace_back(Word{rect, text, conf});

                printf("text: '%s';  \tconf: %.2f; BoundingBox: %d,%d,%d,%d;\n",
                       text, conf, x1, y1, x2, y2);
            }

            delete[] text;
        } while (ri->Next(level));
    }

    cv::Mat imgDraw;
    cv::cvtColor(img, imgDraw, cv::COLOR_GRAY2BGR);

    drawWords(imgDraw, words, lines);
    cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);

    PIX *m = api->GetThresholdedImage();
    cv::Mat thresholded = pixToMat(m);
    cv::rotate(thresholded, thresholded, cv::ROTATE_90_COUNTERCLOCKWISE);

    cv::imshow("result", imgDraw);
    while (cv::waitKey(0) != 27) {}

    cvv::debugFilter(original, img, CVVISUAL_LOCATION, "preprocessed");
    cvv::debugFilter(img, thresholded, CVVISUAL_LOCATION, "thresholded");
    cvv::finalShow();

    api->End();
    delete api;

    return 0;
}
