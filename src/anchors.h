#include "xtensor/xarray.hpp"

class anchors{
public:
    xt::xarray<float> generate_anchors(xt::xarray<float>& feature_map_sizes, xt::xarray<float>& anchor_sizes, xt::xarray<float>& anchor_ratios);
    
    xt::xarray<float> decode_bbox(xt::xarray<float>& anchors, xt::xarray<float>& raw_outputs, xt::xarray<float> variances = {0.1, 0.1, 0.2, 0.2});

private:    
    // template<typename T> std::vector<float> linspace(T start_in, T end_in, int num_in);

};

