#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xsort.hpp"
#include "nms.h"
using namespace xt;

xarray<int> nms::single_class_non_max_suppression(xarray<float>& bboxes, xarray<float>& confidences, double conf_thresh, double iou_thresh, int keep_top_k){
    /*
    do nms on single class.
    Hint: for the specific class, given the bbox and its confidence,
    1) sort the bbox according to the confidence from top to down, we call this a set
    2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
    3) remove the bbox whose IOU is higher than the iou_thresh from the set,
    4) loop step 2 and 3, util the set is empty.
    :param bboxes: numpy array of 2D, [num_bboxes, 4]
    :param confidences: numpy array of 1D. [num_bboxes]
    :param conf_thresh:
    :param iou_thresh:
    :param keep_top_k:
    :return:
    */
    if(bboxes.shape()[0] == 0){return empty<float>({1});}

    auto conf_keep_idx = from_indices(argwhere(confidences >= conf_thresh));
    xarray<float> bboxes_keep = view(bboxes, keep(conf_keep_idx));
    xarray<float> confidences_keep = view(confidences, keep(conf_keep_idx));

    xindex pick;
    xarray<float> xmin = view(bboxes_keep, all(), range(0,1));
    xarray<float> ymin = view(bboxes_keep, all(), range(1,2));
    xarray<float> xmax = view(bboxes_keep, all(), range(2,3));
    xarray<float> ymax = view(bboxes_keep, all(), range(3,4));

    xarray<float> area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3);
    xarray<int> idxs = argsort(confidences_keep, 0);
    while(idxs.shape()[0] > 0){
        int last = idxs.shape()[0]-1;
        int i = idxs(last); 
        pick.push_back(i);

        if(keep_top_k != -1){
            if(pick.size() >= (size_t)keep_top_k){
                break;
            }
        }

        auto idxs_rest = view(idxs, range(0, last));

        auto xmin_rest = view(xmin, keep(idxs_rest));
        xarray<float> overlap_xmin = maximum(xmin(i), xmin_rest);
        auto ymin_rest = view(ymin, keep(idxs_rest));
        xarray<float> overlap_ymin = maximum(ymin(i), ymin_rest);
        auto xmax_rest = view(xmax, keep(idxs_rest));
        xarray<float> overlap_xmax = maximum(xmax(i), xmax_rest);
        auto ymax_rest = view(ymax, keep(idxs_rest));
        xarray<float> overlap_ymax = maximum(ymax(i), ymax_rest);

        xarray<float> overlap_w = maximum(0, overlap_xmax - overlap_xmin);
        xarray<float> overlap_h = maximum(0, overlap_ymax - overlap_ymin);
        xarray<float> overlap_area = overlap_w * overlap_h;

        auto area_rest = view(area, keep(idxs_rest));
        xarray<float> overlap_ratio = overlap_area / (area_rest + area(i) - overlap_area);

        auto overlap_idx = ravel_indices(argwhere(overlap_ratio > iou_thresh), overlap_ratio.shape());
        xarray<float> need_to_be_deleted_idx = concatenate(xtuple(xarray<int>{last}, overlap_idx), 0);
        xarray<float> idxs_left = view(idxs, drop(need_to_be_deleted_idx));
        idxs = idxs_left;       

    }

    std::vector<xindex> pick_vec = {pick};
    auto pick_idxs = from_indices(pick_vec);     
    xarray<int> final_pick = view(conf_keep_idx, keep(pick_idxs)); 
    const auto& s = final_pick.shape();   
    // std::copy(s.cbegin(), s.cend(), std::ostream_iterator<int>(std::cout , " "));
    return final_pick;

}


