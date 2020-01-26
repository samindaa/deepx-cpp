//
// Created by saminda on 1/25/20.
//

#include "common.h"

#include <torch/torch.h>

namespace deepx {

void load_from_state_dict(std::shared_ptr<torch::nn::Module> tgt, std::shared_ptr<torch::nn::Module> src) {
  std::vector<std::string> missing_keys;
  torch::autograd::GradMode::set_enabled(false);
  const torch::OrderedDict<std::string, torch::Tensor> src_p = src->named_parameters();
  const torch::OrderedDict<std::string, torch::Tensor> src_b = src->named_buffers();
  for (auto& tgt_p : tgt->named_parameters()) {
    const auto& key = tgt_p.key();
    const auto* value =  src_p.find(key);
    if (value != nullptr) {
      tgt_p.value().copy_(*value);
    } else {
      missing_keys.emplace_back(key);
    }
  }

  for (auto& tgt_b : tgt->named_buffers()) {
    const auto& key = tgt_b.key();
    const auto* value =  src_b.find(key);
    if (value != nullptr) {
      tgt_b.value().copy_(*value);
    } else {
      missing_keys.emplace_back(key);
    }
  }

  if (!missing_keys.empty()) {
    std::cout << "Following keys are not available in the src: " << std::endl;
    for (const auto& missing_key : missing_keys) {
      std::cout << "\t" << missing_key << std::endl;
    }
  }
  torch::autograd::GradMode::set_enabled(true);
}

} // namespace deepx