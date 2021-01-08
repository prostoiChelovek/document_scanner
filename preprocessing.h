/**
 * @file preprocessing.h
 * @author Люнгрин Андрей aka prostoichelovek <iam.prostoi.chelovek@gmail.com>
 * @date 08 Jan 2021
 * @copyright MIT License
 */

#ifndef DOCUMENT_SCANNER_PREPROCESSING_H
#define DOCUMENT_SCANNER_PREPROCESSING_H

#include <opencv2/opencv.hpp>
#include <opencv2/cvv.hpp>

#include "utils.h"

void removePen(cv::Mat &img, cv::Mat const &gray);

/**
 *  make "almost white" pixels white
 */
void whiten(cv::Mat &img);

/**
 * @see https://stackoverflow.com/a/33971525/9577873
 */
void sharpen(cv::Mat &img, double sigma, double threshold, double amount);

void preprocess(cv::Mat &img);

#endif //DOCUMENT_SCANNER_PREPROCESSING_H
