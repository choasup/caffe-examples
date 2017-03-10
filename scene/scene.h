#ifndef SCENE_H
#define SCENE_H

//#include <io.h>
#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/single_image_data_layer.hpp"
#include <vector>

using caffe::Blob;
using caffe::Layer;
using caffe::SingleImageDataLayer;
using caffe::Caffe;
using caffe::Net;
using std::string;
using boost::shared_ptr;
using std::ifstream;
using std::vector;

#pragma once
class Scene
{
public:
    Scene();
    ~Scene();
    void SceneUnderstanding(const string& im_name);

    shared_ptr<Net<float> > net_scene;
    int index;
    float score;
    vector<float> scene_attr;
};


#endif // SCENE_H
