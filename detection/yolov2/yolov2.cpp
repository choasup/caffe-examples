#include "yolov2.h"
#include <vector>
#include <math.h>
#include "boost/algorithm/string.hpp"
#include <iostream>

using namespace std;
Yolov2::Yolov2()
{
    Caffe::set_mode(Caffe::CPU);

    //string model_def = "resource/Yolov2/googlenet/gnet_region_googlenet_deploy.prototxt";
    //string model_weights = "resource/Yolov2/googlenet/gnet_yolo_region_iter_480000.caffemodel";

    //string model_def = "resource/Yolov2/darknet/gnet_region_train_darknet_deploy.prototxt";
    //string model_weights = "resource/Yolov2/darknet/gnet_yolo_region_darknet_iter_198000.caffemodel";

    //string model_def = "resource/darknet/yolo-voc-v2-deploy2.prototxt";
    //string model_weights = "resource/darknet/yolo-voc-v2_final.caffemodel";

    string model_def = "resource/darknet_v3/gnet_region_test_darknet_deploy.prototxt";
    string model_weights = "resource/darknet_v3/gnet_yolo_region_darknet_v3_iter_106000.caffemodel";

    net_yolov2 = boost::shared_ptr<Net<float> >(new Net<float>(model_def, caffe::TEST));
    net_yolov2->CopyTrainedLayersFrom(model_weights);
}

inline float sigmoid(float x)
{
  return 1. / (1. + exp(-x));
}

float softmax_region(float* input, int classes)
{
  float sum = 0;
  float large = input[0];

  for (int i = 0; i < classes; ++i){
    if (input[i] > large)
      large = input[i];
  }
  for (int i = 0; i < classes; ++i){
    float e = exp(input[i] - large);
    sum += e;
    input[i] = e;
  }
  //std::cout<<"[";
  for (int i = 0; i < classes; ++i){
    input[i] = input[i] / sum;
    //std::cout<<input[i]<<"\t";
  }
  //std::cout<<"]"<<std::endl;
  return 0;
}

void Yolov2::Yolov2Detection(const string& im_name)
{
    clock_t start, finish;
    start = clock();
    int height = 416;
    int width = 416;
    cout<<im_name<<endl;
    cv::Mat cv_img = cv::imread(im_name);
    int im_h = cv_img.rows;
    int im_w = cv_img.cols;
    cv::Mat cv_new(width, height, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::resize(cv_img, cv_img, cv::Size(416, 416), 0.5, 0.5, CV_INTER_LINEAR);

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

    net_yolov2->blob_by_name("data")->Reshape(1, 3, height, width);
    Blob<float> * input_blobs = net_yolov2->input_blobs()[0];

    switch (Caffe::mode()){
    case Caffe::CPU:
        memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
        break;
    default:
        LOG(FATAL) << "Unknow Caffe mode";
    }

    //	net_rpn->blob_by_name("data")->set_cpu_data(data_buf);
    net_yolov2->ForwardFrom(0);

    const boost::shared_ptr<Blob<float> > result = net_yolov2->blob_by_name("conv_reg");
    //const boost::shared_ptr<Blob<float> > result = net_yolov2->blob_by_name("conv22");

    std::cout << result->shape_string() << std::endl;
    std::cout << result->data_at(0,0,0,0) << " " << result->data_at(0,0,0,1) << std::endl;

    pre_all.clear();
    vector<float> pre;
    int k = 0;
    float obj_score = 0;

    float biases[10] = {1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52};
    Blob<float> swap;
    swap.Reshape(result->num(), result->height()*result->width(), 5, result->channels() / 5);
    float* swap_data = swap.mutable_cpu_data();
    int index = 0;
    for (int b = 0; b < result->num(); ++b)
        for (int h = 0; h < result->height(); ++h)
            for (int w = 0; w < result->width(); ++w)
                for (int c = 0; c < result->channels(); ++c)
                {
                    swap_data[index++] = result->data_at(b,c,h,w);
                    //swap_data[index++] = 0;
                }

    //int p_index =  (7*13+4)*125;

    //swap_data[p_index]=-0.1020;
    //swap_data[p_index+1]=2.0867;
    //swap_data[p_index+2]=1.612;
    //swap_data[p_index+3]=1.0515;
    //swap_data[p_index+4]=1.0;
    //swap_data[p_index+5+11]=100;

    for (int b = 0; b < swap.num(); ++b)
        for (int j = 0; j < 13; ++j)
            for (int i = 0; i < 13; ++i)
                for (int n = 0; n < 5; ++n)
                {
                    int index = b * swap.channels() * swap.height() * swap.width() + (j * 13 + i) * swap.height() * swap.width() + n * swap.width();
                    float x = (i + sigmoid(swap_data[index + 0])) / 13;
                    float y = (j + sigmoid(swap_data[index + 1])) / 13;
                    float w = (exp(swap_data[index + 2])*biases[2*n]) / 13;
                    float h = (exp(swap_data[index + 3])*biases[2*n+1]) / 13;
                    softmax_region(swap_data + index + 5 , 20);

                    float label = 0;
                    float obj_score = sigmoid(swap_data[index + 4]);
                    float cls_score = 0;
                    for (int c = 0; c < 20; ++c){
                        //std::cout<<swap_data[index + 5 + c]<<std::endl;
                        if (cls_score < swap_data[index + 5 + c])
                        {
                            cls_score = swap_data[index + 5 + c];
                            label = c;
                        }
                    }

                    if (cls_score*obj_score < 0.2)
                        continue;
                    //if (obj_score < 0.3)
                    //    continue;
                    std::cout<<"box:"<<x<<" "<<y<<" "<<w<<" "<<h<<" "<< label << " "<< obj_score<<" "<<cls_score<<std::endl;


                    float xmin = x - w/2;
                    xmin = xmin>0?xmin:0;
                    xmin = xmin<1?xmin:1;

                    float ymin = y - h/2;
                    ymin = ymin>0?ymin:0;
                    ymin = ymin<1?ymin:1;

                    float xmax = x + w/2;
                    xmax = xmax>0?xmax:0;
                    xmax = xmax<1?xmax:1;

                    float ymax = y + h/2;
                    ymax = ymax>0?ymax:0;
                    ymax = ymax<1?ymax:1;
                    pre.clear();
                    pre.push_back(xmin*im_w);
                    pre.push_back(ymin*im_h);
                    pre.push_back(xmax*im_w);
                    pre.push_back(ymax*im_h);
                    pre.push_back(label+1);
                    pre.push_back(obj_score * cls_score);

                    //std::cout<<pre[0]<<" "<<pre[1]<<" "<<pre[2]<<" "<<pre[3]<<" "<<pre[4]<<" "<<pre[5]<<std::endl;
                    pre_all.push_back(pre);
                }


    this->nmsv2(0.7);
    finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout<<"Time: "<<duration<<std::endl;
    std::cout<<"YOLO Detection Done!"<<std::endl;
}

void Yolov2::nmsv2(double overlap)
{
    vector<int> vPick(1000);
    int nPick = 0;
    std::cout<<"1"<<std::endl;
    int nSample = pre_all.size();
    if (!nSample > 0)
        return;
    std::cout<<"2"<<std::endl;
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
    std::cout<<"3"<<std::endl;
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
    std::cout<<nPick<<std::endl;
    vector<vector <float> > tmp = pre_all;

    pre_all.clear();
    for (int i = 0; i < nPick; i ++)
    {
        int index = vPick[i];
        pre_all.push_back(tmp[index]);

        //std::cout<<tmp[index][0]<<"\t";
        //std::cout<<tmp[index][1]<<"\t";
        //std::cout<<tmp[index][2]<<"\t";
        //std::cout<<tmp[index][3]<<"\t";
        //std::cout<<tmp[index][4]<<"\t";
        //std::cout<<tmp[index][5]<<std::endl;
    }
}

