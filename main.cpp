#include <iostream>
#include <cstdio>

#include <opencv2/opencv.hpp>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

struct Word {
    cv::Rect rect;
    std::string text;
};

cv::Mat pix8ToMat(Pix *pix) {
    int width = pixGetWidth(pix);
    int height = pixGetHeight(pix);

    cv::Mat mat(cv::Size(width, height), CV_8UC3);

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            l_int32 r, g, b;
            pixGetRGBPixel(pix, x, y, &r, &g, &b);

            cv::Vec3b color(b, g, r);
            mat.at<cv::Vec3b>(cv::Point(x, y)) = color;
        }
    }

    return mat;
}

void drawWords(cv::Mat  &img, std::vector<Word> const &words) {
    for (auto const &word : words) {
        cv::rectangle(img, word.rect, {255, 0, 0}, 2);
    }

    cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);

    for (auto const &word : words) {
        cv::Point textOrig = {word.rect.y, img.size().height - word.rect.x - word.rect.width / 2 + 5};
        cv::putText(img, word.text, textOrig, cv::FONT_HERSHEY_COMPLEX, 0.7,
                    {0, 255, 0}, 1);
    }
}

int main(int argc, char *argv[]) {
    auto *api = new tesseract::TessBaseAPI();
    if (api->Init("/usr/share/tesseract-ocr/5/tessdata", "rus+eng")) {
        api->End();

        std::cerr << "Cannot initialize tesseract" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<Word> words;

    api->SetPageSegMode(tesseract::PSM_AUTO_OSD);

    PIX *pix = pixRead("../data/0/01.png");
    api->SetImage(pix);

    api->Recognize(nullptr);

    tesseract::ResultIterator* ri = api->GetIterator();
    tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
    if (ri != nullptr) {
        do {
            const char* word = ri->GetUTF8Text(level);
            float conf = ri->Confidence(level);
            int x1, y1, x2, y2;
            ri->BoundingBox(level, &x1, &y1, &x2, &y2);

            words.emplace_back(Word{cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), word});

            printf("word: '%s';  \tconf: %.2f; BoundingBox: %d,%d,%d,%d;\n",
                   word, conf, x1, y1, x2, y2);
            delete[] word;
        } while (ri->Next(level));
    }

    cv::Mat img = pix8ToMat(pix);
    drawWords(img, words);
    cv::imshow("img", img);
    while (cv::waitKey(0) != 27) {}

    api->End();
    delete api;

    return 0;
}
