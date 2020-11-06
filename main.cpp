#include <iostream>
#include <cstdio>

#include <opencv2/opencv.hpp>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

int morphElem = 0;
int morphSize = 0;
int operation = cv::MORPH_ERODE;
int secondOperation = cv::MORPH_OPEN;

int sigma = 100, threshold = 100, amount = 100;

cv::Mat gImg;

struct Word {
    cv::Rect rect;
    std::string text;
};

struct Line {
    cv::Point a, b;
};

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

void preprocess(cv::Mat &img) {
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    if (sigma % 2 != 1) sigma += 1;
    double _sigma = sigma / 10.0;
    double _threshold = threshold / 10.0;
    double _amount = amount / 10.0;

    cv::imshow("orig", img);

    // make "almost white" pixels white
    cv::Mat tmpThreshould;
    cv::threshold(img, tmpThreshould, 200, 255, cv::THRESH_BINARY);
    cv::erode(tmpThreshould, tmpThreshould,
               cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25)));
    img.setTo(255, tmpThreshould);

    // sharpen
    // https://stackoverflow.com/a/33971525/9577873
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(), _sigma, _sigma);
    cv::Mat lowContrastMask = abs(img - blurred) < _threshold;
    cv::Mat sharpened = img*(1+_amount) + blurred*(-_amount);
    img.copyTo(sharpened, lowContrastMask);
    img = sharpened;

    //cv::fastNlMeansDenoising(img, img, 10, 7, 21);

    cv::threshold(img, img, 100, 255, cv::THRESH_BINARY);
}

void doStuff(int, void *) {
    cv::Mat img = gImg.clone();
    preprocess(img);

    cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::imshow("img", img);
}

int main(int argc, char *argv[]) {
    auto *api = new tesseract::TessBaseAPI();
    if (api->Init("/usr/share/tesseract-ocr/4.00/tessdata/", "rus+eng")) {
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
    gImg = img;

    cv::namedWindow("params", cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_NORMAL);
    cv::createTrackbar("Element:\n 0: Rect - 1: Cross - 2: Ellipse", "params",
                       &morphElem, 2, doStuff);
    cv::createTrackbar("Kernel size:\n 2n +1", "params",
                   &morphSize, 5, doStuff);
    cv::createTrackbar("Op:", "params",
                   &operation, 7, doStuff);
    cv::createTrackbar("Op2:", "params",
                   &secondOperation, 7, doStuff);

    cv::createTrackbar("sigma:", "params",
                   &sigma, 100, doStuff);
    cv::createTrackbar("threshold:", "params",
                   &threshold, 100, doStuff);
    cv::createTrackbar("amount:", "params",
                   &amount, 100, doStuff);

    preprocess(img);
    //while (cv::waitKey(0) != 27) {}

    //return EXIT_SUCCESS;

    api->SetImage((uchar *) img.data, img.size().width, img.size().height, img.channels(), img.step1());
    api->SetSourceResolution(300);

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
                cv::Point a(rect.x + rect.width / 2, rect.y);
                cv::Point b(rect.x + rect.width / 2, rect.y + rect.height);
                lines.emplace_back(Line{a, b});
            } else if (blockType == PolyBlockType::PT_VERT_LINE) {
                cv::Point a(rect.x, rect.y + rect.height / 2);
                cv::Point b(rect.x + rect.width, rect.y + rect.height / 2);
                lines.emplace_back(Line{a, b});
            } else if (PTIsTextType(blockType)) {
                words.emplace_back(Word{rect, text});

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

    cv::imshow("result", imgDraw);
    cv::imshow("preprocessed", img);
    while (cv::waitKey(0) != 27) {}

    api->End();
    delete api;

    return 0;
}
