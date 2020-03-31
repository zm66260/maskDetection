#include "xtensor/xarray.hpp"

class nms{
    public:
        xt::xarray<int> single_class_non_max_suppression(xt::xarray<float>& y_bboxes, xt::xarray<float>& bbox_max_scores, double conf_thresh = 0.5, double iou_thresh = 0.4, int keep_top_k = -1);
};

