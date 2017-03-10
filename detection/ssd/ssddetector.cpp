#include "ssddetector.h"
#include <vector>
#include <math.h>
#include "boost/algorithm/string.hpp"
#include <iostream>

using namespace std;

SSDdetector::SSDdetector()
{
    Caffe::set_mode(Caffe::CPU);

    string model_def = "resource/SSD/models/VGGNet/coco_imagenet/SSD_300x300/deploy.prototxt";
    string model_weights = "resource/SSD/models/VGGNet/coco_imagenet/SSD_300x300/VGG_coco_imagenet_SSD_300x300_loc_loss_50000_iter_70000.caffemodel";
    //string model_weights = "resource/SSD/models/VGGNet/coco_imagenet/SSD_300x300/VGG_coco_imagenet_SSD_300x300_loc_loss_coco_finetune_iter_60000.caffemodel";

    net_ssd = boost::shared_ptr<Net<float> >(new Net<float>(model_def, caffe::TEST));
    net_ssd->CopyTrainedLayersFrom(model_weights);
}

void SSDdetector::SSDetection(const string& im_name)
{
    int height = 300;
    int width = 300;
    cv::Mat cv_img = cv::imread(im_name);
    int img_height = cv_img.rows;
    int img_width = cv_img.cols;

    cv::Mat cv_new(width, height, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::resize(cv_img, cv_img, cv::Size(300, 300), 0.5, 0.5, CV_INTER_LINEAR);

    for (int h = 0; h < cv_new.rows; ++h)
    {
        for (int w = 0; w < cv_new.cols; ++w)
        {
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[0]) - float(104);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[1]) - float(117);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[2]) - float(123);
        }
    }

    float *data_buf;

    data_buf = new float[height * width * 3];

    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            data_buf[(0 * height + h)*width + w] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[0]);
            data_buf[(1 * height + h)*width + w] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[1]);
            data_buf[(2 * height + h)*width + w] = float(cv_new.at<cv::Vec3f>(cv::Point(w, h))[2]);
        }
    }

    net_ssd->blob_by_name("data")->Reshape(1, 3, height, width);
    Blob<float> * input_blobs = net_ssd->input_blobs()[0];

    switch (Caffe::mode()){
    case Caffe::CPU:
        memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
        break;
    default:
        LOG(FATAL) << "Unknow Caffe mode";
    }

    //	net_rpn->blob_by_name("data")->set_cpu_data(data_buf);
    net_ssd->ForwardFrom(0);

    const boost::shared_ptr<Blob<float> > result = net_ssd->blob_by_name("detection_out");

    std::cout << result->shape_string() << std::endl;

    pre_all.clear();
    vector<float> pre;

    for (int i = 0; i < result->height(); i++ )
    {
            pre.clear();
            if ((result->cpu_data()[i*7+2]) > 0.2)
            {
                int xmin = floor(result->cpu_data()[i*7+3]*img_width); //293
                int ymin = floor(result->cpu_data()[i*7+4]*img_height); //402
                int xmax = floor(result->cpu_data()[i*7+5]*img_width);//404
                int ymax = floor(result->cpu_data()[i*7+6]*img_height); //476

                int label = result->cpu_data()[i*7+1];
                float score = result->cpu_data()[i*7+2];

                pre.push_back(xmin);
                pre.push_back(ymin);

                pre.push_back(xmax);
                pre.push_back(ymax);
                pre.push_back(label);
                pre.push_back(score);
                pre_all.push_back(pre);
           }
    }

   this->nms(0.7);
    std::cout<<"SSD Detection Done!"<<std::endl;
}

void SSDdetector::nms(double overlap)
{
    vector<int> vPick(100);
    int nPick = 0;

    int nSample = pre_all.size();
    if (!nSample > 0)
        return;

    vector<double> vArea(nSample);

    vector<float> p;
    for (int i = 0; i < nSample; i++)
    {
        p = pre_all[i];
        vArea[i] = double(p[2] - p[0] + 1)*(p[3] - p[1] + 1);
        if (vArea[i] < 0)
        {
            cout << "error" << endl;
            return;
        }
    }

    std::multimap<float, int> scores;
    for (int i = 0; i < nSample; i++)
    {
        p = pre_all[i];
        scores.insert(std::pair<float, int>(p[5], i));
    }

    do
    {
        int last = scores.rbegin()->second;

        vPick[nPick] = last;
        nPick += 1;
        vector<float> lastV = pre_all[last];
        for (std::multimap<float, int>::iterator it = scores.begin(); it != scores.end();)
        {
            int it_idx = it->second;
            p = pre_all[it_idx];
            float xx1 = max_s(lastV[0], p[0]);
            float yy1 = max_s(lastV[1], p[1]);
            float xx2 = min_s(lastV[2], p[2]);
            float yy2 = min_s(lastV[3], p[3]);

            double w = max_s(float(0.0), xx2 - xx1 + 1);
            double h = max_s(float(0.0), yy2 - yy1 + 1);

            double ov = w*h / (vArea[last] + vArea[it_idx] - w*h);

            if (ov > overlap)
            {
                scores.erase(it);
                it++;
            }
            else
            {
                it++;
            }
        }
    } while (scores.size() != 0);

    vector<vector <float> > tmp = pre_all;

    pre_all.clear();
    for (int i = 0; i < nPick; i ++)
    {
        int index = vPick[i];
        pre_all.push_back(tmp[index]);

        std::cout<<tmp[index][0]<<"\t";
        std::cout<<tmp[index][1]<<"\t";
        std::cout<<tmp[index][2]<<"\t";
        std::cout<<tmp[index][3]<<"\t";
        std::cout<<tmp[index][4]<<"\t";
        std::cout<<tmp[index][5]<<std::endl;
    }
}
