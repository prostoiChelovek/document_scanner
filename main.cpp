#include <iostream>
#include <cstdio>

#include <opencv2/opencv.hpp>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

int main(int argc, char *argv[]) {
    auto *api = new tesseract::TessBaseAPI();
    if (api->Init("/usr/share/tesseract-ocr/5/tessdata", "rus")) {
        api->End();

        std::cerr << "Cannot initialize tesseract" << std::endl;
        return EXIT_FAILURE;
    }

    api->SetPageSegMode(tesseract::PSM_AUTO_OSD);

    PIX *pix = pixRead("../data/0/01.png");
    // PIX *pix = pixRead("../test.png");
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
            printf("word: '%s';  \tconf: %.2f; BoundingBox: %d,%d,%d,%d;\n",
                   word, conf, x1, y1, x2, y2);
            delete[] word;
        } while (ri->Next(level));
    }

    tesseract::Orientation orientation;
    tesseract::WritingDirection direction;
    tesseract::TextlineOrder order;
    float deskew_angle;

    const tesseract::PageIterator *it = api->AnalyseLayout();
    if (it) {
        it->Orientation(&orientation, &direction, &order, &deskew_angle);
        printf(
                "Orientation: %d\nWritingDirection: %d\nTextlineOrder: %d\n"
                "Deskew angle: %.4f\n",
                orientation, direction, order, deskew_angle);
    } else {
        return EXIT_FAILURE;
    }

    delete it;

    api->End();
    delete api;

    return 0;
}
