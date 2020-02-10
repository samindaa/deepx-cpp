//
// Created by saminda on 1/30/20.
//

#ifndef DEEPX_CPP_RL_CPP_SPINUP_ALGOS_VPG_CORE_H_
#define DEEPX_CPP_RL_CPP_SPINUP_ALGOS_VPG_CORE_H_

#include <torch/torch.h>

namespace deepx::spinup::algos::vpg {

using Function = std::function<torch::Tensor(torch::Tensor)>;

torch::Tensor identity(torch::Tensor x) {
  return x;
}

torch::nn::Sequential mlp(const std::vector<int> &sizes, Function activation, Function output_activation = identity) {
  torch::nn::Sequential seq;
  for (auto j = 0; j < sizes.size() - 1; ++j) {
    seq->push_back(torch::nn::Linear(sizes[j], sizes[j + 1]));
    seq->push_back(j < sizes.size() - 2 ? torch::nn::Functional(activation) : torch::nn::Functional(output_activation));
  }
  return seq;
}

class Actor : public torch::nn::Module {
 public:
  virtual torch::Tensor _distribution(torch::Tensor obs) = 0;

  virtual torch::Tensor _log_prob_from_distribution(torch::Tensor pi, torch::Tensor act) = 0;

  torch::Tensor forward(torch::Tensor obs) {
    auto pi = _distribution(obs);
    return pi;
  }
};

class MLPCategoricalActor : public Actor {
 public:
  MLPCategoricalActor(int obs_dim, int act_dim, const std::vector<int> &hidden_sizes, Function activation) {
    std::vector<int> sizes;
    sizes.push_back(obs_dim);
    sizes.insert(sizes.end(), hidden_sizes.begin(), hidden_sizes.end());
    sizes.push_back(act_dim);
    logits_net_ = register_module("logits_net", mlp(sizes, activation));
  }

  torch::Tensor _distribution(torch::Tensor obs) {
    auto logits = logits_net_->forward(obs);
    return {};
  }

 protected:
  torch::nn::Sequential logits_net_{nullptr};
};


} // namespace deepx::spinup::algos::vpg

#endif //DEEPX_CPP_RL_CPP_SPINUP_ALGOS_VPG_CORE_H_
