#include "detector.h"
#include <vector>
#include <math.h>
#include "boost/algorithm/string.hpp"

using namespace std;


struct Info
{
    float score;
    const float* head;
};

bool compare(const Info& Info1, const Info& Info2)
{
    return Info1.score > Info2.score;
}


Detector::~Detector()
{
}

Detector::Detector()
{
    Caffe::set_mode(Caffe::CPU);

    clock_t start, finish;
    start = clock();

    string model_file_rpn = "resource/detection/faster_rcnn_VOC0712_vgg_16layers/proposal_test.prototxt";
    string weights_file_rpn = "resource/detection/faster_rcnn_VOC0712_vgg_16layers/proposal_final";

    string model_file_det = "resource/detection/faster_rcnn_VOC0712_vgg_16layers/detection_test.prototxt";
    string weights_file_det = "resource/detection/faster_rcnn_VOC0712_vgg_16layers/detection_final";

    net_rpn = boost::shared_ptr<Net<float> >(new Net<float>(model_file_rpn, caffe::TEST));
    net_rpn->CopyTrainedLayersFrom(weights_file_rpn);

    net_det = boost::shared_ptr<Net<float> >(new Net<float>(model_file_det, caffe::TEST));
    net_det->CopyTrainedLayersFrom(weights_file_det);
    finish = clock();

    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout << "Load NET Time = " << duration << "s" << endl;

    Pre_classes.insert(pair<int, string>(0, "aeroplane"));
    Pre_classes.insert(pair<int, string>(1, "bicycle"));
    Pre_classes.insert(pair<int, string>(2, "bird"));
    Pre_classes.insert(pair<int, string>(3, "boat"));
    Pre_classes.insert(pair<int, string>(4, "bottle"));
    Pre_classes.insert(pair<int, string>(5, "bus"));
    Pre_classes.insert(pair<int, string>(6, "car"));
    Pre_classes.insert(pair<int, string>(7, "cat"));
    Pre_classes.insert(pair<int, string>(8, "chair"));
    Pre_classes.insert(pair<int, string>(9, "cow"));

    Pre_classes.insert(pair<int, string>(10, "dinningtable"));
    Pre_classes.insert(pair<int, string>(11, "dog"));
    Pre_classes.insert(pair<int, string>(12, "horse"));
    Pre_classes.insert(pair<int, string>(13, "motorbike"));
    Pre_classes.insert(pair<int, string>(14, "person"));
    Pre_classes.insert(pair<int, string>(15, "pottedplant"));
    Pre_classes.insert(pair<int, string>(16, "sheep"));
    Pre_classes.insert(pair<int, string>(17, "sofa"));
    Pre_classes.insert(pair<int, string>(18, "train"));
    Pre_classes.insert(pair<int, string>(19, "tvmonitor"));
}

void Detector::RegionProposal(const string& im_name)
{
    //cv::Mat cv_img = cv::Mat(375, 500, CV_8UC3, cv::Scalar(128, 128, 128));
    //cout << "E = " << endl << " " << im << endl << endl;
    cv::Mat cv_img = cv::imread(im_name);

    clock_t start, finish;
    start = clock();

    cv::Mat cv_new(cv_img.rows, cv_img.cols, CV_32FC3, cv::Scalar(0, 0, 0));

    for (int h = 0; h < cv_img.rows; ++h)
    {
        for (int w = 0; w < cv_img.cols; ++w)
        {
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[0]) - float(102.9801);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[1]) - float(115.9465);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[2]) - float(122.7717);
        }
    }

    //	cv::namedWindow("Example", CV_WINDOW_AUTOSIZE);
    //	cv::imshow("Example", cv_new);


    const int  max_input_side = 1000;
    const int  min_input_side = 600;

    int max_side = max(cv_img.rows, cv_img.cols);
    int min_side = min(cv_img.rows, cv_img.cols);

    float max_side_scale = float(max_input_side) / float(max_side);
    float min_side_scale = float(min_input_side) / float(min_side);
    float max_scale = min(max_side_scale, min_side_scale);

    img_scale = max_scale;

    int height = int(cv_img.rows * img_scale);
    int width = int(cv_img.cols * img_scale);

    cv::Mat cv_resized;
    cv::resize(cv_new, cv_resized, cv::Size(width, height), 0.5, 0.5, cv::INTER_LINEAR);

    //	cout << "E = " << endl << " " << cv_new << endl << endl;

    float *data_buf;
    data_buf = new float[height * width * 3];

    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            data_buf[(0 * height + h)*width + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[0]);
            data_buf[(1 * height + h)*width + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[1]);
            data_buf[(2 * height + h)*width + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[2]);
        }
    }

    net_rpn->blob_by_name("data")->Reshape(1, 3, height, width);
    Blob<float> * input_blobs = net_rpn->input_blobs()[0];


    switch (Caffe::mode()){
    case Caffe::CPU:
        memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
        break;
    default:
        LOG(FATAL) << "Unknow Caffe mode";
    }

    //	net_rpn->blob_by_name("data")->set_cpu_data(data_buf);
    net_rpn->ForwardFrom(0);

    //*******************************************************************************************
    finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout << "RPN Time = " << duration << "s" << endl;
    start = clock();
    //*******************************************************************************************

    const float* data_bbox_delt;
    const float* data_bbox_prob;

    vector<Blob<float>*> out;

    data_bbox_delt = net_rpn->blob_by_name("proposal_bbox_pred")->cpu_data();
    data_bbox_prob = net_rpn->blob_by_name("proposal_cls_prob")->cpu_data();

    conv = net_rpn->blob_by_name("conv5_3");
    const float *conv5_3 = conv->cpu_data();

//	cout << conv->shape_string() << endl;

    //for (int u = 0; u < conv->num(); u++)
    //	for (int v = 0; v < conv->channels(); v++)
    //		for (int w = 0; w < conv->height(); w++)
    //			for (int x = 0; x < conv->width(); x++)
    //				cout << conv->data_at(u, v, w, x) << endl;


    //	cout << data_bbox_delt[0] << '\t' << data_bbox_delt[1] << '\t' <<data_bbox_delt[2] << "\t" << data_bbox_delt[3] << endl;


    out = net_rpn->output_blobs();
//	cout << out.size() << endl;

    Blob<float> *b1;
    Blob<float> *b2;

    //	Blob<float> *b = net_->blob_by_name("conv5_3");

    b1 = out.at(0);
    b2 = out.at(1);

    //cout << "Size:" << b1->shape_string() << endl;
    //cout << "Size:" << b2->shape_string() << endl;		//1*36*38*50

    //cout << b2->data_at(0, 1, 0, 0) << "\t" << b2->data_at(0, 1, 38, 0) << "\t" << b2->data_at(0, 1, 76, 0) << "\t" << b2->data_at(0, 1, 1, 0) << endl;
    //cout << b2->data_at(0, 1, 39, 0) << "\t" << b2->data_at(0, 1, 77, 0) << endl;

    //	cout << b1->width() << endl;	//50
    //	cout << b1->height() << endl;	//38
    //	cout << b1->channels() << endl;	//36
    //	cout << b1->num() << endl;		//1

    //b1->height();
    //b1->width();
    //	cout << b1->data_at(0, 0, 0, 0) << "\t" << b1->data_at(0, 0, 0, 1) << "\t" << b1->data_at(0, 0, 0, 2) << "\t" << b1->data_at(0, 0, 0, 3) << endl;
    //	cout << b1->data_at(0, 0, 0, 4) << "\t" << b1->data_at(0, 0, 0, 5) << "\t" << b1->data_at(0, 0, 0, 6) << "\t" << b1->data_at(0, 0, 0, 7) << endl;

    int num = b1->channels()*b1->height()*b1->width()*b1->num();
    //	b1->Reshape(b1->num(),b1->width(),b1->height(),b1->channels());

    //	memcpy(b1->mutable_cpu_data(), data_bbox_delt, sizeof(float) * b1->count());

    //cout << b1->data_at(0, 0, 0, 0) << "\t" << b1->data_at(0, 0, 0, 1) << "\t" << b1->data_at(0, 0, 0, 2) << "\t" << b1->data_at(0, 0, 0, 3) << endl;
    //cout << b1->data_at(0, 0, 0, 4) << "\t" << b1->data_at(0, 0, 0, 5) << "\t" << b1->data_at(0, 0, 0, 6) << "\t" << b1->data_at(0, 0, 0, 7) << endl;

    //b1->Reshape(1, 1, 4, num / 4);

    //cout << "Size:" << b1->shape_string() << endl;		//1*1*4*17100

    float *pos = new float[num];

    int index = 0;
    for (int u = 0; u < b1->num(); u++)
        for (int v = 0; v < b1->width(); v++)
            for (int w = 0; w < b1->height(); w++)
                for (int x = 0; x < b1->channels(); x++)
                    pos[index++] = b1->data_at(u, x, w, v);

    cv::Mat box_deltas = cv::Mat(num / 4, 4, CV_32FC1, pos);
    //std::cout << "row = " << box_deltas.row(0) << std::endl;
    //std::cout << "row = " << box_deltas.row(9) << std::endl;
    //std::cout << "row = " << box_deltas.row(17) << std::endl;
    //std::cout << "row = " << box_deltas.row(18) << std::endl;

    float *data_scores = new float[num / 4];
    index = 0;
    for (int u = 0; u < b2->num(); u++)
        for (int x = 1; x < 2; x++)
            for (int v = 0; v < b2->width(); v++)
                for (int w = 0; w < b2->height(); w++)
                    data_scores[index++] = b2->data_at(u, x, (w % 9) * 38 + w / 9, v);

    cv::Mat scores = cv::Mat(num / 4, 1, CV_32FC1, data_scores);
    //std::cout << "scores=" << scores.row(0) << endl;
    //std::cout << "scores=" << scores.row(10) << endl;
    //std::cout << "scores=" << scores.row(20) << endl;
    //std::cout << "scores=" << scores.row(17100 - 2) << endl;
    //std::cout << "scores=" << scores.row(17100 - 1) << endl;
    //************************************************** ȡ\B3\F6\C1\CBblob\D6е\C4\CA\FD\BE\DD box_deltas and scores ********************************************************//


    int out_size_height = ceil(height*1.0 / 16);
    int out_size_width = ceil(width*1.0 / 16);

    //cout << out_size_height << "\t" << out_size_width << endl;
    //cout << pos[0] << "\t" << pos[1] << "\t" << pos[2] << "\t" << pos[3] << "\t" << pos[4] << endl;

    vector<int> t_x, t_y;								//\C9\FA\B3\C9meshgrid
    for (int i = 0; i < out_size_width; i++)
        t_x.push_back(i * 16);

    for (int j = 0; j < out_size_height; j++)
        t_y.push_back(j * 16);

    cv::Mat X, Y, shift_X, shift_Y;
    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);

    //	std::cout << "X = " << X.col(1) << std::endl;
    //<< "Y = " << Y.col(0) << std::endl;
    //	std::cout << "X = " << X.size() << std::endl;
    //	std::cout << "Y = " << Y.size() << std::endl;

    shift_X = X.reshape(0, X.rows*X.cols);
    shift_Y = Y.reshape(0, Y.rows*Y.cols);

    //	std::cout << "X = " << shift_X.row(1) << std::endl;
    //	std::cout << "Y = " << shift_Y.row(2) << std::endl;

    //	std::cout << "X = " << shift_X.size() << std::endl;
    //	std::cout << "Y = " << shift_Y.size() << std::endl;

    cv::Mat anchors_XY(shift_X.rows, 4, shift_X.type());
    //cv::Mat submat = anchors_XY.colRange(0, 1);
    shift_X.copyTo(anchors_XY.colRange(0, 1));
    shift_Y.copyTo(anchors_XY.colRange(1, 2));
    shift_X.copyTo(anchors_XY.colRange(2, 3));
    shift_Y.copyTo(anchors_XY.colRange(3, 4));

    //************************************************************** 1900*4\B5\C4meshgrid *********************************************************//

    float data_anchors[36] = { -83, -39, 100, 56,
        -175, -87, 192, 104,
        -359, -183, 376, 200,
        -55, -55, 72, 72,
        -119, -119, 136, 136,
        -247, -247, 264, 264,
        -35, -79, 52, 96,
        -79, -167, 96, 184,
        -167, -343, 184, 360 };

    cv::Mat anchors = cv::Mat(9, 4, CV_32FC1, data_anchors);

    //cout << "E = " << endl << " " << anchors << endl << endl;

    cv::Mat real_anchors = cv::Mat(anchors_XY.rows * 9, anchors_XY.cols, CV_32FC1);

    for (int i = 0; i < real_anchors.rows; i++)
    {
        anchors_XY.row(i / 9).copyTo(real_anchors.rowRange(i, i + 1));
    }

    //std::cout << "row = " << real_anchors.row(0) << std::endl;
    //std::cout << "row = " << real_anchors.row(9) << std::endl;
    //std::cout << "row = " << real_anchors.row(17100-2) << std::endl;
    //std::cout << "row = " << real_anchors.row(17100-1) << std::endl;

    cv::Mat dst;
    for (int i = 0; i < real_anchors.rows; i = i + 9)
    {
        cv::add(real_anchors.rowRange(i, i + 9), anchors, dst);
        dst.copyTo(real_anchors.rowRange(i, i + 9));
    }



    //real anchore:17100*4  box_deltas:17100*4
    cv::Mat src_w = cv::Mat(real_anchors.rows, 1, CV_32FC1);
    cv::Mat src_h = cv::Mat(real_anchors.rows, 1, CV_32FC1);
    cv::Mat src_ctr_x = cv::Mat(real_anchors.rows, 1, CV_32FC1);
    cv::Mat src_ctr_y = cv::Mat(real_anchors.rows, 1, CV_32FC1);

    src_w = real_anchors.colRange(2, 3) - real_anchors.colRange(0, 1) + 1;
    src_h = real_anchors.colRange(3, 4) - real_anchors.colRange(1, 2) + 1;
    src_ctr_x = real_anchors.colRange(0, 1) + 0.5*(src_w - 1);
    src_ctr_y = real_anchors.colRange(1, 2) + 0.5*(src_h - 1);

    cv::Mat dst_ctr_x = cv::Mat(box_deltas.rows, 1, CV_32FC1);
    cv::Mat dst_ctr_y = cv::Mat(box_deltas.rows, 1, CV_32FC1);
    cv::Mat dst_scl_x = cv::Mat(box_deltas.rows, 1, CV_32FC1);
    cv::Mat dst_scl_y = cv::Mat(box_deltas.rows, 1, CV_32FC1);

    dst_ctr_x = box_deltas.colRange(0, 1);
    dst_ctr_y = box_deltas.colRange(1, 2);
    dst_scl_x = box_deltas.colRange(2, 3);
    dst_scl_y = box_deltas.colRange(3, 4);

    cv::Mat pred_ctr_x = cv::Mat(box_deltas.rows, 1, CV_32FC1);
    cv::Mat pred_ctr_y = cv::Mat(box_deltas.rows, 1, CV_32FC1);
    cv::Mat pred_w = cv::Mat(box_deltas.rows, 1, CV_32FC1);
    cv::Mat pred_h = cv::Mat(box_deltas.rows, 1, CV_32FC1);


    cv::multiply(dst_ctr_x, src_w, pred_ctr_x);
    pred_ctr_x = pred_ctr_x + src_ctr_x;

    cv::multiply(dst_ctr_y, src_h, pred_ctr_y);
    pred_ctr_y = pred_ctr_y + src_ctr_y;

    cv::exp(dst_scl_x, dst_scl_x);
    cv::exp(dst_scl_y, dst_scl_y);

    cv::multiply(dst_scl_x, src_w, pred_w);
    cv::multiply(dst_scl_y, src_h, pred_h);

    cv::Mat pred_boxes = cv::Mat(box_deltas.rows, box_deltas.cols, CV_32FC1);

    pred_boxes.colRange(0, 1) = pred_ctr_x - 0.5*(pred_w - 1);
    pred_boxes.colRange(1, 2) = pred_ctr_y - 0.5*(pred_h - 1);
    pred_boxes.colRange(2, 3) = pred_ctr_x + 0.5*(pred_w - 1);
    pred_boxes.colRange(3, 4) = pred_ctr_y + 0.5*(pred_h - 1);

    //**************************************** \CA\E4\B3\F6\CB\F9\D3е\C4pred_boxes maybe\D3\D0һЩ\BB\E1\D3\D0\CE\F3\B2\EE **************************************************//


    float data_im_size[4] = { cv_img.cols - 1, cv_img.rows - 1, cv_img.cols - 1, cv_img.rows - 1 };
    float data_scaled_im_size[4] = { 1.0 / (width - 1), 1.0 / (height - 1), 1.0 / (width - 1), 1.0 / (height - 1) };

    cv::Mat im_size = cv::Mat(1, 4, CV_32FC1, data_im_size);
    cv::Mat scaled_im_size = cv::Mat(1, 4, CV_32FC1, data_scaled_im_size);

    cv::Mat ratio = cv::Mat(1, 4, CV_32FC1);
    cv::multiply(im_size, scaled_im_size, ratio);

    //	cout << "E = " << endl << " " << ratio << endl << endl;
    pred_boxes = pred_boxes - 1;
    for (int i = 0; i < pred_boxes.rows; i++)
    {
        cv::multiply(pred_boxes.row(i), ratio, pred_boxes.row(i));
    }
    //	cv::multiply(pred_boxes, ratio, pred_boxes);
    pred_boxes = pred_boxes + 1;


    //cout << "E = " << endl << " " << ratio << endl << endl;

    //std::cout << "pred_boxes = " << pred_boxes.size() << std::endl;
    //std::cout << "pred_boxes = " << pred_boxes.row(0) << std::endl;
    //std::cout << "pred_boxes = " << pred_boxes.row(1) << std::endl;
    //std::cout << "pred_boxes = " << pred_boxes.row(2) << std::endl;
    //std::cout << "pred_boxes = " << pred_boxes.row(3) << std::endl;
    //std::cout << "pred_boxes = " << pred_boxes.row(10) << std::endl;
    //std::cout << "pred_boxes = " << pred_boxes.row(20) << std::endl;
    //std::cout << "pred_boxes = " << pred_boxes.row(17100-1) << std::endl;

    //cv::waitKey(0);
    //cv_img.release();
    //cv::destroyWindow("Example");
    //cv::min(pred_boxes.col(0),cv_img.cols)
    cv::min(pred_boxes.col(0), cv_img.cols, pred_boxes.col(0));
    cv::max(pred_boxes.col(0), 1, pred_boxes.col(0));

    cv::min(pred_boxes.col(1), cv_img.rows, pred_boxes.col(1));
    cv::max(pred_boxes.col(1), 1, pred_boxes.col(1));

    cv::min(pred_boxes.col(2), cv_img.cols, pred_boxes.col(2));
    cv::max(pred_boxes.col(2), 1, pred_boxes.col(2));

    cv::min(pred_boxes.col(3), cv_img.rows, pred_boxes.col(3));
    cv::max(pred_boxes.col(3), 1, pred_boxes.col(3));

    //**************************************** \CA\E4\B3\F6\CB\F9\D3е\C4pred_boxes(17100*4) scores(17100*1) Mat**************************************************//

    //	std::cout << "pred_boxes = " << pred_boxes.row(0) << std::endl;
    int *flag = new int[pred_boxes.rows];
    int sum = 0;
    for (int i = 0; i < pred_boxes.rows; i++)
    {
        //float f1 = cvGetReal2D(pred_boxes,i,0);
        //float f1 = pred_boxes.at<cv::Vec3f>(cv::Point(0, i))[0];
        //float f2 = pred_boxes.at<cv::Vec3f>(cv::Point(1, i))[1];
        //float f3 = pred_boxes.at<cv::Vec3f>(cv::Point(2, i))[2];
        //float f4 = pred_boxes.at<cv::Vec3f>(cv::Point(3, i))[3];

        float f1 = pred_boxes.at<float>(i, 0);
        float f2 = pred_boxes.at<float>(i, 1);
        float f3 = pred_boxes.at<float>(i, 2);
        float f4 = pred_boxes.at<float>(i, 3);

        //		std::cout << "boxes = " << pred_boxes.row(i) << std::endl;
        //ratio = pred_boxes.row(i);
        if (f3 - f1 + 1 > 15 && f4 - f2 + 1 > 15)
        {
            flag[i] = 1;
            sum++;
        }
        else
            flag[i] = 0;
    }

    //	cout << sum << endl;
    cv::Mat boxes = cv::Mat(sum, 5, CV_32FC1);

    //	for (int i = 0; i < scores.rows; i++)
    //		std::cout << "scores=" << i << "\t" << scores.row(i) << endl;

    index = 0;
    for (int i = 0; i < pred_boxes.rows; i++)
    {
        if (flag[i] == 1)
        {
            boxes.at<float>(index, 0) = pred_boxes.at<float>(i, 0);
            boxes.at<float>(index, 1) = pred_boxes.at<float>(i, 1);
            boxes.at<float>(index, 2) = pred_boxes.at<float>(i, 2);
            boxes.at<float>(index, 3) = pred_boxes.at<float>(i, 3);
            boxes.at<float>(index, 4) = scores.at<float>(i, 0);

            //	std::cout << "boxes = " << boxes.row(index) << std::endl;
            index++;
        }
    }

    //boxes \CA\C7û\D3\D0\CE\CA\CC\E2\B5ģ\ACscoresҲ\CA\C7û\D3\D0\CE\CA\CC\E2\B5\C4


    //sort \BB\B9û\D3\D0ʵ\CF\D6 \C4\C5~~~~~~
    //cv::Mat tmp = cv::Mat(1, 5, CV_32FC1);
    //for (int i = 0; i < boxes.rows; i++)
    //	for (int j = 0; j < boxes.rows - i - 1; j++)
    //	{
    //		if (boxes.at<float>(j, 4) < boxes.at<float>(j + 1, 4))
    //		{
    //			tmp = boxes.row(j);
    //			boxes.row(j) = boxes.row(j + 1);
    //			boxes.row(j + 1) = tmp;
    //		}
    //	}

    //std::cout << "boxes = " << boxes.row(0) << std::endl;
    //std::cout << "boxes = " << boxes.row(1) << std::endl;
    //std::cout << "boxes = " << boxes.row(2) << std::endl;
    //std::cout << "boxes = " << boxes.row(3) << std::endl;
    //std::cout << "boxes = " << boxes.row(4) << std::endl;
    //*******************************************

    int per_nms_topN = 6000;
    float nms_overlap_thres = 0.7;
    int after_nms_topN = 300;


    cv::Mat sortIdxArr = cv::Mat(sum, 5, CV_32S);
    cv::sortIdx(boxes, sortIdxArr, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);

    num = min_s(per_nms_topN, boxes.rows);
    cv::Mat aboxes = cv::Mat(num, 5, CV_32FC1);

    index = 0;
    for (int i = 0; i < num; i++)
    {
        index = sortIdxArr.at<int>(i, 4);
        //		std::cout << "sort = " << sortIdxArr.row(i) << std::endl;

        boxes.row(index).copyTo(aboxes.row(i));
        //	std::cout << "boxes = " << boxes.row(index) << std::endl;
        //	std::cout << "aboxes = " << aboxes.row(i) << std::endl;
    }


    vector<int> vPick(num);
    int nPick = 0;
    nms(aboxes, nms_overlap_thres, vPick, nPick);


    //	for (int i = 0; i < nPick; i++)
    //		std::cout << vPick[i] << endl;

    //cout << num << endl;

    //	std::cout << "aboxes = " << aboxes.size() << std::endl;
    //	std::cout << "aboxes = " << aboxes.rowRange(0,5) << std::endl;
    //	std::cout << "aboxes = " << aboxes.row(aboxes.rows-1) << std::endl;

    num = min_s(after_nms_topN, nPick);
    real_boxes = cv::Mat(num, 5, CV_32FC1);

    for (int i = 0; i < num; i++)
    {
        aboxes.row(vPick[i]).copyTo(real_boxes.row(i));
        //cout << real_boxes.row(i) << endl;
    }

    //cout << "real_boxes =" << real_boxes.size() << endl;

    //	for (int i = 0; i < 10; i++)
    //		cout << conv5_3[i] << endl;

    cv::Mat rois_blob = cv::Mat(real_boxes.rows, 5, CV_32S);

    Blob<float> rois;

    rois.Reshape(real_boxes.rows, 5, 1, 1);
    float *data_rois = rois.mutable_cpu_data();

    for (int i = 0; i < real_boxes.rows; i++)
    {
        rois_blob.at<int>(i, 0) = 1;
        data_rois[i * 5 + 0] = rois_blob.at<int>(i, 0) - 1;

        rois_blob.at<int>(i, 1) = round((real_boxes.at<float>(i, 0) - 1)*img_scale) + 1;
        data_rois[i * 5 + 1] = rois_blob.at<int>(i, 1) - 1;

        rois_blob.at<int>(i, 2) = round((real_boxes.at<float>(i, 1) - 1)*img_scale) + 1;
        data_rois[i * 5 + 2] = rois_blob.at<int>(i, 2) - 1;

        rois_blob.at<int>(i, 3) = round((real_boxes.at<float>(i, 2) - 1)*img_scale) + 1;
        data_rois[i * 5 + 3] = rois_blob.at<int>(i, 3) - 1;

        rois_blob.at<int>(i, 4) = round((real_boxes.at<float>(i, 3) - 1)*img_scale) + 1;
        data_rois[i * 5 + 4] = rois_blob.at<int>(i, 4) - 1;

        //		cout << rois_blob.row(i) << endl;
    }


    //*******************************************************************************************
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout << "Data Process Time = " << duration << "s" << endl;
    //system("pause");

    start = clock();
    //*******************************************************************************************


    net_det->blob_by_name("data")->Reshape(conv->num(), conv->channels(), conv->height(), conv->width());
    net_det->blob_by_name("rois")->Reshape(rois.num(), rois.channels(), rois.height(), rois.width());


    //net_rpn->blob_by_name("data")->Reshape(1, 3, height, width);

    Blob<float> * input_blobs_1 = net_det->input_blobs()[0];
    Blob<float> * input_blobs_2 = net_det->input_blobs()[1];

    //switch (Caffe::mode()){
    //case Caffe::CPU:
    memcpy(input_blobs_1->mutable_cpu_data(), conv5_3, sizeof(float) * input_blobs_1->count());
    memcpy(input_blobs_2->mutable_cpu_data(), data_rois, sizeof(float) * input_blobs_2->count());
    //	break;
    //default:
    //	LOG(FATAL) << "Unknow Caffe mode";
    //}

    ////	cout << net_->blob_by_name("data")->data_at(0, 0, 0, 0) << "\t" << net_->blob_by_name("data")->data_at(0, 0, 0, 1) << "\t" << net_->blob_by_name("data")->data_at(0, 0, 0, 2) << "\t" << net_->blob_by_name("data")->data_at(0, 0, 0, 3) << endl;
    ////	cout << net_->blob_by_name("data")->data_at(0, 0, 0, 0) << "\t" << net_->blob_by_name("data")->data_at(0, 1, 0, 0) << "\t" << net_->blob_by_name("data")->data_at(0, 2, 0, 0) << "\t" << net_->blob_by_name("data")->data_at(0, 0, 1, 0) << endl;


    //	net_det->blob_by_name("data")->set_cpu_data(conv5_3);
    //	net_det->blob_by_name("rois")->set_cpu_data(data_rois);

    vector<Blob<float>*> output;
    Blob<float>* box;
    Blob<float>* s;
    net_det->ForwardFrom(0);

    output = net_det->output_blobs();

    box = output.at(0);
    s = output.at(1);

    //cout << "box =" << box->shape_string() << endl;
    //cout << "s =" << s->shape_string() << endl;
    //cout << "num =" << box->num() << endl;
    //cout << "channels =" << box->channels() << endl;
    //cout << "height =" << box->height() << endl;
    //cout << "width = " << box->width() << endl;

    //for (int u = 0; u < s->num(); u++)
    //	for (int v = 0; v < s->channels(); v++)
    //		for (int w = 0; w < s->height(); w++)
    //			for (int x = 0; x < s->width(); x++)
    //				cout << s->data_at(u, v, w, x) << endl;


    cv::Mat Mat_box = cv::Mat(box->num(), box->channels(), CV_32FC1);
    cv::Mat Mat_s = cv::Mat(s->num(), s->channels(), CV_32FC1);


    for (int i = 0; i < Mat_box.rows; i++)
    {
        for (int j = 0; j < Mat_box.cols; j++)
            Mat_box.at<float>(i, j) = box->data_at(i, j, 0, 0);

//		cout << Mat_box.row(i) << endl;

        for (int j = 0; j < Mat_s.cols; j++)
            Mat_s.at<float>(i, j) = s->data_at(i, j, 0, 0);

        //cout << Mat_s.row(i) << endl;

    }


    cv::Mat *temp;
    num = Mat_box.rows;
    vector<cv::Mat*> pre;

    float fwidth, fheight, ctr_x, ctr_y, dx, dy, dw, dh, fpred_ctr_x, fpred_ctr_y, fpred_w, fpred_h;

    for (int i = 0; i < num; i++)
    {
        fwidth = real_boxes.at<float>(i, 2) - real_boxes.at<float>(i, 0) + 1;
        fheight = real_boxes.at<float>(i, 3) - real_boxes.at<float>(i, 1) + 1;
        ctr_x = real_boxes.at<float>(i, 0) + 0.5 * fwidth;
        ctr_y = real_boxes.at<float>(i, 1) + 0.5 * fheight;

        temp = new cv::Mat(21, 5, CV_32FC1);
        for (int j = 0; j < 21; j++)
        {
            dx = Mat_box.at<float>(i, 4 * j + 0);
            dy = Mat_box.at<float>(i, 4 * j + 1);
            dw = Mat_box.at<float>(i, 4 * j + 2);
            dh = Mat_box.at<float>(i, 4 * j + 3);

            fpred_ctr_x = ctr_x + fwidth * dx;
            fpred_ctr_y = ctr_y + fheight * dy;
            fpred_w = fwidth * exp(dw);
            fpred_h = fheight * exp(dh);


            temp->at<float>(j, 0) = max_s(min_s(fpred_ctr_x - 0.5* fpred_w, cv_img.cols - 1), 0);
            temp->at<float>(j, 1) = max_s(min_s(fpred_ctr_y - 0.5* fpred_h, cv_img.rows - 1), 0);
            temp->at<float>(j, 2) = max_s(min_s(fpred_ctr_x + 0.5* fpred_w, cv_img.cols - 1), 0);
            temp->at<float>(j, 3) = max_s(min_s(fpred_ctr_y + 0.5* fpred_h, cv_img.rows - 1), 0);
            temp->at<float>(j, 4) = Mat_s.at<float>(i, j);
        }
//		cout << "Temp = " << endl << " " << *temp << endl << endl;

        pre.push_back(temp);

    //	cout << "Temp = " << endl << " " << *pre.back() << endl << endl;


    }

    //cout << pre.size() << endl;

    cv::Mat *every_class;
    vector<cv::Mat*> classes;
    int class_num = 20;

    temp = new cv::Mat(21, 5, CV_32FC1);
    for (int i = 0; i < class_num; i++)
    {
        every_class = new cv::Mat(num, 5, CV_32FC1);
        for (int j = 0; j < num; j++)
        {
            //temp = cv::Mat(21, 5, CV_32FC1);
            //temp = pre[j];
            //cout << "Temp = " << endl << " " << pre.back() << endl << endl;
            temp = pre[j];
//			cout << "Temp = " << endl << " " << *pre[j] << endl << endl;
            every_class->at<float>(j, 0) = temp->at<float>(i + 1, 0);
            every_class->at<float>(j, 1) = temp->at<float>(i + 1, 1);
            every_class->at<float>(j, 2) = temp->at<float>(i + 1, 2);
            every_class->at<float>(j, 3) = temp->at<float>(i + 1, 3);
            every_class->at<float>(j, 4) = temp->at<float>(i + 1, 4);
        }
        classes.push_back(every_class);
    }

    cv::Mat* boxS;
    float thres = 0.7;
    vector<cv::Mat*> result;

    for (int i = 0; i < classes.size(); i++)
    {
        every_class = new cv::Mat(num, 5, CV_32FC1);

        every_class = classes[i];
        //cout << *every_class << endl;

        vector<int> vPick(300);
        int nPick = 0;
        nms(*every_class, 0.3, vPick, nPick);

        index = 0;
        for (int j = 0; j < nPick; j++)
        {
            if (every_class->at<float>(vPick[j], 4) > thres)
            {
                //cout << every_class->row(vPick[j]) << endl;
                boxS = new cv::Mat(1, 6, CV_32FC1);
                boxS->at<float>(0, 0) = every_class->at<float>(vPick[j], 0);
                boxS->at<float>(0, 1) = every_class->at<float>(vPick[j], 1);
                boxS->at<float>(0, 2) = every_class->at<float>(vPick[j], 2);
                boxS->at<float>(0, 3) = every_class->at<float>(vPick[j], 3);
                boxS->at<float>(0, 4) = every_class->at<float>(vPick[j], 4);
                boxS->at<float>(0, 5) = i;
                result.push_back(boxS);
                index++;
            }
        }
    }

    fstream fin("detection_result.txt",ios::out|ios::app);
    fin<<"Region Proposals: \n";

    int id_box = 0;
    boxS = new cv::Mat(1, 6, CV_32FC1);
    for (int i = 0; i < result.size(); i++)
    {
        boxS = result[i];
        int c = (int)boxS->at<float>(0, 5);

        fin<<id_box++<<"\t";
        fin<<boxS->at<float>(0,0)<<"\t";
        fin<<boxS->at<float>(0,1)<<"\t";
        fin<<boxS->at<float>(0,2)<<"\t";
        fin<<boxS->at<float>(0,3)<<"\t";
        fin<<c<<"\t";
        fin<<Pre_classes[c]<<"\n";

        std::cout<<id_box<<"\t";
        std::cout<<im_name<<"\t";
        std::cout<<boxS->at<float>(0,0)<<"\t";
        std::cout<<boxS->at<float>(0,1)<<"\t";
        std::cout<<boxS->at<float>(0,2)<<"\t";
        std::cout<<boxS->at<float>(0,3)<<"\t";
        std::cout<<c<<"\t";
        std::cout<<Pre_classes[c]<<"\n";


        //cv::rectangle(cv_img, cv::Point(boxS->at<float>(0, 0), boxS->at<float>(0, 1)), cv::Point(boxS->at<float>(0, 2), boxS->at<float>(0, 3)), cv::Scalar((c * 20) % 255, (c * 40) % 255, (c * 80) % 255), 2);
        //CvFont font;

        //cvInitFont(&font, CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0);
        //cv::putText(cv_img, Pre_classes[c], cv::Point(boxS->at<float>(0, 0) + 10, boxS->at<float>(0, 1) + 12), 1, 1, cv::Scalar((c * 20) % 255, (c * 40) % 255, (c * 80) % 255), 1);
        //cv::PutText(src, showMsg, cvPoint(200, 200), &font, CV_RGB(255, 0, 0));
        //cout << *boxS << endl;
    }
    fin<<"\n\n";
    fin.close();

    //*******************************************************************************************
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout << "Detection Time = " << duration << "s" << endl;

    start = clock();
    //*******************************************************************************************

    cout << "Detection Done!" << endl;
    //cv::imshow(im_name, cv_img);
    //cv::waitKey(0);
}

void Detector::nms(cv::Mat &input_boxes, double overlap, std::vector<int> &vPick, int &nPick)
{
    int nSample = input_boxes.rows;
    int nDim_boxes = input_boxes.cols;

    vector<double> vArea(nSample);

    for (int i = 0; i < nSample; i++)
    {
        vArea[i] = double(input_boxes.at<float>(i, 2) - input_boxes.at<float>(i, 0) + 1)*(input_boxes.at<float>(i, 3) - input_boxes.at<float>(i, 1) + 1);
        if (vArea[i] < 0)
        {
            cout << "error" << endl;
            return;
        }
    }

    std::multimap<float, int> scores;
    for (int i = 0; i < nSample; i++)
        scores.insert(std::pair<float, int>(input_boxes.at<float>(i, 4), i));

    do
    {
        int last = scores.rbegin()->second;

        vPick[nPick] = last;
        nPick += 1;

        for (std::multimap<float, int>::iterator it = scores.begin(); it != scores.end();)
        {
            int it_idx = it->second;
            float xx1 = max_s(input_boxes.at<float>(last, 0), input_boxes.at<float>(it_idx, 0));
            float yy1 = max_s(input_boxes.at<float>(last, 1), input_boxes.at<float>(it_idx, 1));
            float xx2 = min_s(input_boxes.at<float>(last, 2), input_boxes.at<float>(it_idx, 2));
            float yy2 = min_s(input_boxes.at<float>(last, 3), input_boxes.at<float>(it_idx, 3));

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
}
