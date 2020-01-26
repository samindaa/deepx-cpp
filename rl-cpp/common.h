//
// Created by saminda on 1/25/20.
//

#ifndef DEEPX_CPP_RL_CPP_COMMON_H_
#define DEEPX_CPP_RL_CPP_COMMON_H_

#include <torch/torch.h>
#include <torch/csrc/api/include/torch/nn/module.h>

namespace deepx {

void load_from_state_dict(std::shared_ptr<torch::nn::Module> tgt, std::shared_ptr<torch::nn::Module> src);

} // namespace deepx

#endif //DEEPX_CPP_RL_CPP_COMMON_H_
