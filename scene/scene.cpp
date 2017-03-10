#include "scene.h"
#include <vector>
#include <cblas.h>
#include "caffe/util/math_functions.hpp"

using namespace std;
using namespace caffe;

Scene::Scene()
{
    Caffe::set_mode(Caffe::CPU);

    std::string proto_file_net;
    std::string model_file_net;
    char temp_string[256];

    sprintf(temp_string, "resource/placesCNN/places205CNN_deploy.prototxt");
    proto_file_net = temp_string;

    sprintf(temp_string, "resource/placesCNN/places205CNN_iter_300000.caffemodel");
    model_file_net = temp_string;

    net_scene = boost::shared_ptr<Net<float> >(new Net<float>(proto_file_net, caffe::TEST));
    net_scene->CopyTrainedLayersFrom(model_file_net);
}

Scene::~Scene()
{
}

void Scene::SceneUnderstanding(const string& im_name)
{
    int height = 227;
    int width = 227;
    cv::Mat cv_img = cv::imread(im_name);
    cv::Mat cv_new(width, height, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::resize(cv_img, cv_img, cv::Size(227, 227), 0.5, 0.5, CV_INTER_LINEAR);

    for (int h = 0; h < cv_new.rows; ++h)
    {
        for (int w = 0; w < cv_new.cols; ++w)
        {
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[0]) - float(110.9801);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[1]) - float(117.9465);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[2]) - float(117.7717);
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

    net_scene->blob_by_name("data")->Reshape(1, 3, height, width);
    Blob<float> * input_blobs = net_scene->input_blobs()[0];

    switch (Caffe::mode()){
    case Caffe::CPU:
        memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
        break;
    default:
        LOG(FATAL) << "Unknow Caffe mode";
    }

    //	net_rpn->blob_by_name("data")->set_cpu_data(data_buf);
    net_scene->ForwardFrom(0);

    const boost::shared_ptr<Blob<float> > result = net_scene->blob_by_name("prob");
    const boost::shared_ptr<Blob<float> > fc7 = net_scene->blob_by_name("fc7");
    //cout << fc7->shape_string() << endl;

    Blob<float> w;
    BlobProto w_file;
    ReadProtoFromBinaryFile("resource/placesCNN/w_sceneAttr.blob", &w_file);
    w.FromProto(w_file, true);
    //cout << w.shape_string() << endl;

    Blob<float> sceneAttr;
    sceneAttr.Reshape(1,1,1,102);


    const float* f_p = fc7->cpu_data(); //1*4096 A 4096*1
    const float* w_p = w.cpu_data(); //102*4096 B
    float* s_p = sceneAttr.mutable_cpu_data();//1*102 C

    caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, 102, 1, 4096, 1, w_p, f_p, 0, s_p);
    //caffe_cpu_gemm<float>(CblasNoTrans, CblasTrans, 1, 102, 4096, 1, w_p, f_p, 0, s_p);

    scene_attr.clear();
    for (int i = 0; i < sceneAttr.width(); i ++)
    {
        scene_attr.push_back(sceneAttr.data_at(0,0,0,i));
        //if (scene_attr[i] > 0)
            //cout<< scene_attr[i]<<"\t";
    }

    int dim_features = result->count();
    float* feature_blob_data;
    feature_blob_data = result->mutable_cpu_data();

    float max = feature_blob_data[0];

    for (int i = 0; i < dim_features; i++){ //get the max score and the scene index
        if (max < feature_blob_data[i])
        {
            max = feature_blob_data[i];
            index = i;
            score = max;
        }
    }

    cv_img.release();
    cv_new.release();
    //std::cout<<"scene:"<<index<<" "<<max<<endl;
    //cout << index << "\t" << max << endl;
    std::cout<<"Scene Understanding Done!"<<std::endl;
}
