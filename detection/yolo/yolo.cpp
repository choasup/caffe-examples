#include "yolo.h"
#include <vector>
#include <math.h>
#include "boost/algorithm/string.hpp"
#include <iostream>

using namespace std;
Yolo::Yolo()
{
    Caffe::set_mode(Caffe::CPU);

    string model_def = "resource/Yolo/gnet_deploy.prototxt";
    string model_weights = "resource/Yolo/gnet_yolo_iter_32000_others.caffemodel";

    net_yolo = boost::shared_ptr<Net<float> >(new Net<float>(model_def, caffe::TEST));
    net_yolo->CopyTrainedLayersFrom(model_weights);
}

void Yolo::YoloDetection(const string& im_name)
{
    clock_t start, finish;
    start = clock();
    int height = 448;
    int width = 448;
    cout<<im_name<<endl;
    cv::Mat cv_img = cv::imread(im_name);
    int im_h = cv_img.rows;
    int im_w = cv_img.cols;
    cv::Mat cv_new(width, height, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::resize(cv_img, cv_img, cv::Size(448, 448), 0.5, 0.5, CV_INTER_LINEAR);

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

    net_yolo->blob_by_name("data")->Reshape(1, 3, height, width);
    Blob<float> * input_blobs = net_yolo->input_blobs()[0];

    switch (Caffe::mode()){
    case Caffe::CPU:
        memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
        break;
    default:
        LOG(FATAL) << "Unknow Caffe mode";
    }

    //	net_rpn->blob_by_name("data")->set_cpu_data(data_buf);
    net_yolo->ForwardFrom(0);

    const boost::shared_ptr<Blob<float> > result = net_yolo->blob_by_name("regression");

    std::cout << result->shape_string() << std::endl;

    pre_all.clear();
    vector<float> pre;
    int k = 0;
    float obj_score = 0;
    for (int i = 0; i < 49; i ++)
    {
        for (int k = 0; k < 2; k ++)
        {
            int index_k = 49 * (20 + k) + i;
            if (result->data_at(0,index_k,0,0) < 0)
                continue;
            obj_score = result->data_at(0,index_k,0,0);


        //int index_1 = 49 * 20 + i;
        //int index_2 = 49 * 21 + i;

        //float obj_score = 0;
        //if (result->data_at(0,index_1,0,0)>result->data_at(0,index_2,0,0))
        //{
        //    k = 0;
          //  obj_score = result->data_at(0,index_1,0,0);
        //}
        //else
        //{
          //  k = 1;
            //obj_score = result->data_at(0,index_2,0,0);
       // }
        int pos_index = 49*(22+k*4)+i;
        int pos_x = pos_index;
        int pos_y = pos_index + 49;
        int pos_w = pos_index + 49 * 2;
        int pos_h = pos_index + 49 * 3;

        int cls_label = 0;
        float cls_score = result->data_at(0,i,0,0);

        for (int j = 0;j < 20; j ++)
            if (cls_score < result->data_at(0,i + j * 49,0,0))
            {
                cls_label = j;
                cls_score = result->data_at(0,i + j * 49,0,0);
            }

        std::cout<<cls_label<<"\t";
        std::cout<<cls_score*obj_score<<std::endl;

        float x = (i%7+result->data_at(0,pos_x,0,0))/7*im_w;
        float y = (i/7+result->data_at(0,pos_y,0,0))/7*im_h;
        float w = pow(result->data_at(0,pos_w,0,0),2)*im_w;
        float h = pow(result->data_at(0,pos_h,0,0),2)*im_h;
        if (cls_score*obj_score>0.3){
            pre.clear();
            float x_min = x-w/2;
            x_min = x_min>0?x_min:0;

            float y_min = y-h/2;
            y_min = y_min>0?y_min:0;

            float x_max = x+w/2;
            x_max = x_max>im_w?im_w:x_max;

            float y_max = y+h/2;
            y_max = y_max>im_h?im_h:y_max;

            pre.push_back(x_min);
            pre.push_back(y_min);
            pre.push_back(x_max);
            pre.push_back(y_max);
            pre.push_back(cls_label+1);
            pre.push_back(cls_score*obj_score);
            pre_all.push_back(pre);
        }
        }
    }
    this->nms(0.7);
    finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout<<"Time: "<<duration<<std::endl;
    std::cout<<"YOLO Detection Done!"<<std::endl;
}

void Yolo::nms(double overlap)
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

