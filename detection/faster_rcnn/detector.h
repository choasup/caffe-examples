#ifndef DETECTOR_H
#define DETECTOR_H

#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "boost/algorithm/string.hpp"
//#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using caffe::Blob;
using caffe::Layer;
using caffe::Caffe;
using caffe::Net;
using std::string;
using boost::shared_ptr;
using std::ifstream;

#pragma once

#define max_s(a, b) (((a)>(b)) ? (a) :(b))
#define min_s(a, b) (((a)<(b)) ? (a) :(b))

const int class_num = 2;

class Detector
{
public:
    Detector();
    ~Detector();

    //Detector(const string& model_file, const string& trained_file);
    void RegionProposal(const string& im_name);
    // Detection(const string& im_name);
    void nms(cv::Mat &input_boxes, double overlap, std::vector<int> &vPick, int &nPick);

private:
    shared_ptr<Net<float> > net_rpn;
    shared_ptr<Net<float> > net_det;
    cv::Mat real_boxes;
    float img_scale;
    //const float* conv5_3;
    shared_ptr<Blob<float> > conv;
    std::map<int, string> Pre_classes;

};

#endif // DETECTOR_H
