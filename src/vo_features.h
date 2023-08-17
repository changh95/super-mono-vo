/*

The MIT License

Copyright (c) 2015 Avi Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include <memory>
#include "super_glue.h"
#include "super_point.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

void super_featureTracking(std::shared_ptr<SuperGlue> &superglue,
                           Eigen::Matrix<double, 259, Eigen::Dynamic> &feature_points0,
                           Eigen::Matrix<double, 259, Eigen::Dynamic> &feature_points1, std::vector<cv::Point2f> &points1,
                           std::vector<cv::Point2f> &points2, std::vector<uchar> &status) {
    std::vector<cv::DMatch> superglue_matches;
    superglue->matching_points(feature_points0, feature_points1, superglue_matches);
}

void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2, std::vector<uchar> &status) {

//this function automatically gets rid of points for which tracking fails

    std::vector<float> err;
    const auto winSize = cv::Size(21, 21);
    const auto termcrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++) {
        const auto pt = points2.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
            if ((pt.x < 0) || (pt.y < 0)) {
                status.at(i) = 0;
            }
            points1.erase(points1.begin() + (i - indexCorrection));
            points2.erase(points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }

    }

}

void super_featureDetection(std::shared_ptr<SuperPoint> &superpoint, cv::Mat img, std::vector<cv::Point2f> &points,
                            Eigen::Matrix<double, 259, Eigen::Dynamic> &feature_points) {
    superpoint->infer(img, feature_points);

    for (int i = 0; i < feature_points.cols(); ++i) {
        const auto pt = cv::Point2f(feature_points(1, i), feature_points(2, i));
        points.push_back(pt);
    }
}

void featureDetection(cv::Mat img_1,
                      std::vector<cv::Point2f> &points1) {   //uses FAST as of now, modify parameters as necessary
    std::vector<cv::KeyPoint> keypoints_1;
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    cv::FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
    cv::KeyPoint::convert(keypoints_1, points1, std::vector<int>());
}
