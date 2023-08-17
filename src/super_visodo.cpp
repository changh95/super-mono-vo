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

#include "vo_features.h"
#include "read_config.h"
#include "super_glue.h"
#include "super_point.h"


#define MAX_FRAME 4541
#define MIN_NUM_FEAT 2000

// IMP: Change the file directories (4 places) according to where your dataset is saved before running!

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal) {
    std::string line;
    int i = 0;
    std::ifstream myfile("/data/00/00.txt");
    double x = 0, y = 0, z = 0;
    double x_prev, y_prev, z_prev;
    if (myfile.is_open()) {
        while ((getline(myfile, line)) && (i <= frame_id)) {
            z_prev = z;
            x_prev = x;
            y_prev = y;
            std::istringstream in(line);
            //cout << line << '\n';
            for (int j = 0; j < 12; j++) {
                in >> z;
                if (j == 7) y = z;
                if (j == 3) x = z;
            }

            i++;
        }
        myfile.close();
    } else {
        std::cout << "Unable to open file" << std::endl;
        return 0;
    }

    return std::sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev));

}


int main(int argc, char **argv) {
    const std::string config_path = "../config.yaml";
    const std::string model_dir = "../superpointglue/weights";

    Configs configs(config_path, model_dir);

    std::cout << "Building inference engine......" << std::endl;
    auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
    if (!superpoint->build()) {
        std::cerr << "Error in SuperPoint building engine. Please check your onnx model path." << std::endl;
        return 0;
    }
    auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);
    if (!superglue->build()) {
        std::cerr << "Error in SuperGlue building engine. Please check your onnx model path." << std::endl;
        return 0;
    }

    cv::Mat R_f, t_f; //the final rotation and translation vectors containing the

    std::ofstream myfile;
    myfile.open("results1_2.txt");

    double scale = 1.00;

    char text[100];
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(10, 50);

    //TODO: add a function to load these values directly from KITTI's calib files
    // WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
    double focal = 718.8560;
    cv::Point2d pp(607.1928, 185.2157);

    char filename1[100];
    char filename2[100];

    sprintf(filename1, "/data/00/image_0/%06d.png", 0);
    sprintf(filename2, "/data/00/image_0/%06d.png", 1);

    cv::Mat img1 = cv::imread(filename1);
    cv::Mat img2 = cv::imread(filename2);

    cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> points1, points2;        //vectors to store the coordinates of the feature points
    super_detect_and_match(superpoint, superglue, img1, img2, points1, points2);

    cv::Mat E, R, t, mask;
    E = cv::findEssentialMat(points2, points1, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
    cv::recoverPose(E, points2, points1, R, t, focal, pp, mask);

    R_f = R.clone();
    t_f = t.clone();

    clock_t begin = clock();

    cv::namedWindow("Road facing camera", cv::WINDOW_AUTOSIZE);// Create a window for display.
    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);// Create a window for display.

    cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);

    for (int numFrame = 1; numFrame < MAX_FRAME; numFrame++) {
        sprintf(filename1, "/data/00/image_0/%06d.png", numFrame);
        sprintf(filename2, "/data/00/image_0/%06d.png", numFrame + 1);

        img1 = cv::imread(filename1);
        img2 = cv::imread(filename2);

        cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img2, img2, cv::COLOR_BGR2GRAY);

        points1.clear();
        points2.clear();
        super_detect_and_match(superpoint, superglue, img1, img2, points1, points2);

        E = cv::findEssentialMat(points2, points1, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
        cv::recoverPose(E, points2, points1, R, t, focal, pp, mask);

        scale = getAbsoluteScale(numFrame + 2, 0, t.at<double>(2));

        if ((scale > 0.1) && (t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {
            t_f = t_f + scale * (R_f * t);
            R_f = R * R_f;

        } else {
            //cout << "scale below 0.1, or incorrect translation" << endl;
        }

        // lines for printing results (KITTI)
        myfile << R_f.at<double>(0, 0) << " " << R_f.at<double>(0, 1) << " " << R_f.at<double>(0, 2) << " "
               << t_f.at<double>(0) << " "
               << R_f.at<double>(1, 0) << " " << R_f.at<double>(1, 1) << " " << R_f.at<double>(1, 2) << " "
               << t_f.at<double>(1) << " "
               << R_f.at<double>(2, 0) << " " << R_f.at<double>(2, 1) << " " << R_f.at<double>(2, 2) << " "
               << t_f.at<double>(2) << std::endl;

        int x = int(t_f.at<double>(0)) + 300;
        int y = int(t_f.at<double>(2)) + 100;
        cv::circle(traj, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 2);

        cv::rectangle(traj, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), cv::FILLED);
        sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1),
                t_f.at<double>(2));
        cv::putText(traj, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);


        cv::imshow("Road facing camera", img1);
        cv::imshow("Trajectory", traj);

        cv::waitKey(1);
    }

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Total time taken: " << elapsed_secs << "s" << std::endl;

    //cout << R_f << endl;
    //cout << t_f << endl;

    return 0;
}
