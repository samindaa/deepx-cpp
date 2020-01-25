//
// Created by saminda on 12/31/19.
//

#ifndef DEEPX_CPP_PTAN_CPP_ACTIONS_H_
#define DEEPX_CPP_PTAN_CPP_ACTIONS_H_

#include <memory>
#include <torch/torch.h>

// Abstract class which converts scores to the actions.
class ActionSelector {
 public:
  virtual torch::Tensor operator()(torch::Tensor scores) = 0;
};

// Selects actions using argmax.
class ArgmaxActionSelector : public ActionSelector {
 public:
  torch::Tensor operator()(torch::Tensor scores) override {
    return torch::argmax(scores, /*dim=*/1);
  }
};

class EpsilonGreedyActionSelector : public ActionSelector {
 public:
  explicit EpsilonGreedyActionSelector(double epsilon = 0.05,
                                       torch::Device device = torch::kCPU,
                                       std::shared_ptr<ActionSelector> selector = nullptr)
      : epsilon_(epsilon), device_(device), selector_(selector ? selector : std::make_shared<ArgmaxActionSelector>()) {}

  torch::Tensor operator()(torch::Tensor scores) override {
    auto sizes = scores.sizes();
    auto batch_size = sizes[0], n_actions = sizes[1];
    auto actions = (*selector_)(scores);
    auto mask = (torch::rand(batch_size).to(device_) < epsilon_).nonzero_numpy()[0];
    if (mask.numel() > 0) {
      actions.index_put_(mask, /*values=*/
                         torch::multinomial(/*self=*/torch::randperm(n_actions).to(device_), /*num_samples=*/
                                                     mask.numel()));
    }
    return actions;
  }

 private:
  double epsilon_;
  torch::Device device_;
  std::shared_ptr<ActionSelector> selector_;

};

// ProbabilityActionSelector
// EpsilonTracker

#endif //DEEPX_CPP_PTAN_CPP_ACTIONS_H_
