//
// Created by saminda on 1/24/20.
//

#ifndef DEEPX_CPP_RL_CPP_DQN_H_
#define DEEPX_CPP_RL_CPP_DQN_H_

#include <torch/torch.h>

namespace deepx {

class Dqn : public torch::nn::Module {
 public:
  Dqn(int64_t num_inputs, int64_t num_actions) {
    layaers_ = register_module("layers", torch::nn::Sequential(
        torch::nn::Linear(num_inputs, 128),
        torch::nn::Functional(torch::relu),
        torch::nn::Linear(128, 128),
        torch::nn::Functional(torch::relu),
        torch::nn::Linear(128, num_actions)));
  }

  torch::Tensor forward(torch::Tensor x) {
    return layaers_->forward(x);
  }

  torch::Tensor act(torch::Tensor state) {
    torch::Tensor q_value = forward(state);
    torch::Tensor action = std::get<1>(q_value.max(1));
    return action;
  }

 private:
  torch::nn::Sequential layaers_{nullptr};
};

} // namespace deepx


#endif //DEEPX_CPP_RL_CPP_DQN_H_
