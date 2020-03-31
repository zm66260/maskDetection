#include <vector>
#include "anchors.h"
#include "xtensor/xpad.hpp"
#include "xtensor/xadapt.hpp"

using namespace std;
using namespace xt;


xarray<float> anchors::generate_anchors(xt::xarray<float>& feature_map_sizes, xt::xarray<float>& anchor_sizes, xt::xarray<float>& anchor_ratios){
  /*
    generate anchors.
    param feature_map_sizes: list of list, for example: [[40,40], [20,20]]
    param anchor_sizes: list of list, for example: [[0.05, 0.075], [0.1, 0.15]]
    param anchor_ratios: list of list, for example: [[1, 0.5], [1, 0.5]]
    param offset: default to 0.5
    return:
  */
    
    xarray<float> anchor_bboxes = zeros<float>({1,4});
    for(unsigned int i=0; i < feature_map_sizes.shape()[0]; i++){
      xarray<float> cx = (linspace<float>(0, feature_map_sizes(i, 0) - 1, (int)feature_map_sizes(i, 0)) + 0.5) / feature_map_sizes(i, 0);
      xarray<float> cy = (linspace<float>(0, feature_map_sizes(i, 1) - 1, (int)feature_map_sizes(i, 1)) + 0.5) / feature_map_sizes(i, 1);
      // xtuple<xarray<float>, xarray<float>> gird = meshgrid(cx, cy);
      auto grid = meshgrid(cx, cy);
      xarray<float> cx_grid = get<0>(grid);
      xarray<float> cy_grid = get<1>(grid);
      xarray<float> cx_grid_exp = expand_dims(cx_grid, 2);
      xarray<float> cy_grid_exp = expand_dims(cy_grid, 2);
      xarray<float> center = concatenate(xtuple(cx_grid_exp, cy_grid_exp), 2);

      int num_anchors = row(anchor_sizes, i).shape()[0] + row(anchor_ratios, i).shape()[0] - 1;  
      xarray<float> center_tiled = tile(center, {1, 1, 2 * num_anchors}); 
      vector<float> anchor_width_heights;

      xarray<float> anchor_size = row(anchor_sizes, i);
      xarray<float> anchor_ratio = row(anchor_ratios, i);
      for(unsigned int j = 0; j < anchor_size.shape()[0]; j++){
        float scale = anchor_size(j);
        float ratio = anchor_ratio(0);
        float width = scale * sqrt(ratio);
        float height = scale / sqrt(ratio);
        anchor_width_heights.push_back(-width / 2.0);
        anchor_width_heights.push_back(-height / 2.0);
        anchor_width_heights.push_back(width / 2.0);
        anchor_width_heights.push_back(height / 2.0);
      }

      for(unsigned int j = 1; j < anchor_ratio.shape()[0]; j++){
        float ratio = anchor_ratio(j);
        float s1 = anchor_size(0);
        float width = s1 * sqrt(ratio);
        float height = s1 / sqrt(ratio);
        anchor_width_heights.push_back(-width / 2.0);
        anchor_width_heights.push_back(-height / 2.0);
        anchor_width_heights.push_back(width / 2.0);
        anchor_width_heights.push_back(height / 2.0);
      }

      vector<int> shape = {num_anchors * 4};
      auto anchors = adapt(anchor_width_heights, shape);
      xarray<float> anchors_broadcast = broadcast(anchors, center_tiled.shape());
      xarray<float> bbox_coords = center_tiled + anchors_broadcast;
      xarray<float> bbox_coords_reshape = bbox_coords.reshape({-1, 4});
      anchor_bboxes = concatenate(xtuple(anchor_bboxes, bbox_coords_reshape), 0); 
    }
    anchor_bboxes = view(anchor_bboxes, drop(0), all());
    return anchor_bboxes;
}


xarray<float> anchors::decode_bbox(xt::xarray<float>& anchors, xt::xarray<float>& raw_outputs, xt::xarray<float> variances){
/*
    Decode the actual bbox according to the anchors.
    the anchor value order is:[xmin,ymin, xmax, ymax]
    :param anchors: xarray with shape [batch, num_anchors, 4]
    :param raw_outputs: xarray with the same shape with anchors
    :param variances: list of float, default=[0.1, 0.1, 0.2, 0.2]
    :return:
*/

    xarray<float> anchor_centers_x = (view(anchors, all(), all(), range(0, 1)) + view(anchors, all(), all(), range(2, 3))) / 2;
    xarray<float> anchor_centers_y = (view(anchors, all(), all(), range(1, 2)) + view(anchors, all(), all(), range(3, 4))) / 2;
    xarray<float> anchors_w = view(anchors, all(), all(), range(2, 3)) - view(anchors, all(), all(), range(0, 1));
    xarray<float> anchors_h = view(anchors, all(), all(), range(3, 4)) - view(anchors, all(), all(), range(1, 2));
    xarray<float> raw_outputs_rescale = raw_outputs * variances;
    xarray<float> predict_center_x = view(raw_outputs_rescale, all(), all(), range(0, 1)) * anchors_w + anchor_centers_x;
    xarray<float> predict_center_y = view(raw_outputs_rescale, all(), all(), range(1, 2)) * anchors_h + anchor_centers_y;
    xarray<float> predict_w = exp(view(raw_outputs_rescale, all(), all(), range(2, 3))) * anchors_w;
    xarray<float> predict_h = exp(view(raw_outputs_rescale, all(), all(), range(3, 4))) * anchors_h;
    // xarray<float> predict_xmin = predict_center_x - predict_w / 2;
    // xarray<float> predict_ymin = predict_center_y - predict_h / 2;
    // xarray<float> predict_xmax = predict_center_x + predict_w / 2;
    // xarray<float> predict_ymax = predict_center_y + predict_h / 2;
    xarray<float> predict_ymin = predict_center_x - predict_h / 2;
    xarray<float> predict_xmin = predict_center_y - predict_w / 2;
    xarray<float> predict_ymax = predict_center_x + predict_h / 2;
    xarray<float> predict_xmax = predict_center_y + predict_w / 2;        
    xarray<float> predict_bbox = concatenate(xtuple(predict_xmin, predict_ymin, predict_xmax, predict_ymax), 2);
    return predict_bbox;

}










