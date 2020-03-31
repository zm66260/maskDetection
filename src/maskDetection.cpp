#include<opencv2/opencv.hpp>
#include<torch/script.h>
#include "anchors.h"
#include "nms.h"
#include "xtensor/xsort.hpp"
#include "xtensor/xview.hpp"
#include<iostream>
#include<memory>
#include<string>


using namespace std;
using namespace xt;

enum class id2class{
    Mask,
    NoMask
};

int main(){
    cv::VideoCapture stream(0);
    cv::namedWindow("Mask Detection", cv::WINDOW_AUTOSIZE);

    torch::jit::script::Module module;
    module = torch::jit::load("../model/face_mask_detection_libtorch.pth");
    module.to(at::kCUDA);

    cv::Mat frame;
    cv::Mat image;
    cv::Mat input;   

    xt::xarray<float> feature_map_sizes = {{33, 33}, {17, 17}, {9, 9}, {5, 5}, {3, 3}};
    xt::xarray<float> anchor_sizes = {{0.04, 0.056}, {0.008, 0.11}, {0.16, 0.22}, {0.32, 0.45}, {0.64, 0.72}};
    xt::xarray<float> anchor_ratios = {{1.0, 0.62, 0.42}, {1.0, 0.62, 0.42},{1.0, 0.62, 0.42}, {1.0, 0.62, 0.42}, {1.0, 0.62, 0.42}};

    nms infer_nsm;
    anchors infer_anchors;
    xt::xarray<float> anchor_bboxes = infer_anchors.generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios);
   
    xarray<float> anchors_exp = expand_dims(anchor_bboxes, 0);

    xarray<float> y_bboxes_output = empty<float>(anchors_exp.shape());
    vector<size_t> shape = {anchors_exp.shape()[0], anchors_exp.shape()[1], 2};
    xarray<float> y_cls_output = empty<float>(shape);  

    while(1){
        stream >> frame;
        int width = frame.cols;
        int height = frame.rows;
        resize(frame,image, {260, 260});

        imshow("resize image", image);
        cv::cvtColor(image, input, cv::COLOR_BGR2RGB);

        torch::Tensor tensor_image = torch::from_blob(input.data, {1, input.rows, input.cols, 3}, torch::kByte);
        tensor_image = tensor_image.permute({0,3,1,2});
        tensor_image = tensor_image.toType(torch::kFloat);
        tensor_image = tensor_image.div(255);
        tensor_image = tensor_image.to(torch::kCUDA);   

        // torch::Tensor result = module.forward({tensor_image}).toTensor();
        auto result = module.forward({tensor_image}).toTuple()->elements();
        torch::Tensor y_bboxes_tensor = result[0].toTensor();
        torch::Tensor y_cls_tensor = result[1].toTensor();
        float* y_bboxes_ptr = y_bboxes_tensor.to(torch::kCPU).data<float>();
        float* y_cls_ptr = y_cls_tensor.to(torch::kCPU).data<float>();

        for(unsigned int i = 0; i < anchors_exp.shape()[0]; i++){
            for(unsigned int j = 0; j < anchors_exp.shape()[1]; j++){
                for(unsigned int k = 0; k < anchors_exp.shape()[2]; k++){
                    y_bboxes_output(i, j, k) = (*y_bboxes_ptr++);
                }
                for(unsigned int k = 0; k < shape[2]; k++){
                    // y_cls_output(i, j, k) = y_cls[i][j][k].item().to<float>();
                    y_cls_output(i, j, k) = (*y_cls_ptr++);
                }                
            }
        }

        xarray<float> y_bboxes = squeeze(infer_anchors.decode_bbox(anchors_exp, y_bboxes_output));        
        xarray<float> y_cls = squeeze(y_cls_output);
    
        xarray<float> bbox_max_scores = amax(y_cls, 1);
        xarray<int> bbox_max_score_classes = argmax(y_cls, 1);

        // keep_idx is the alive bounding box after nms.
        xarray<int> keep_idxs = infer_nsm.single_class_non_max_suppression(y_bboxes, bbox_max_scores);

        int indx = keep_idxs.shape()[0];
        cv::Point* pt1 = new cv::Point[indx];
        cv::Point* pt2 = new cv::Point[indx];
        cv::Scalar* color = new cv::Scalar[indx];
        vector<string>label;
        cout << indx << endl;
       
        for(int i = 0; i < indx; i++){
            int idx = keep_idxs(i);
            // float conf = bbox_max_scores(idx);
            int class_id = bbox_max_score_classes(idx);
            xarray<float> bbox = view(y_bboxes, idx, all());
            // std::copy(bbox.cbegin(), bbox.cend(), std::ostream_iterator<float>(std::cout , " "));            
            int xmin = max(0, int(bbox(0) * width));
            int ymin = max(0, int(bbox(1) * height));
            int xmax = min(int(bbox(2) * width), width);
            int ymax = min(int(bbox[3] * height), height);

            if(class_id == 0){
                color[i] = cv::Scalar(0, 255, 0);
                label.push_back("Mask");
            }
            else{
                color[i] = cv::Scalar(0, 0, 255);
                label.push_back("NoMask");
            }
            pt1[i] = cv::Point(xmin, ymin);
            pt2[i] = cv::Point(xmax, ymax);        

            cv::rectangle(frame, pt1[i], pt2[i], color[i], 2);
            cv::putText(frame, label[i], {pt1[i].x + 2, pt1[i].y - 2}, cv::FONT_HERSHEY_SIMPLEX, 0.8, color[i]); 

        }

        delete [] pt1;
        delete [] pt2;
        delete [] color;

        imshow("Mask Detection", frame);
        cv::waitKey(1);

    }
    
}

