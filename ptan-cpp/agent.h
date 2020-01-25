//
// Created by saminda on 1/1/20.
//

#ifndef DEEPX_CPP_PTAN_CPP_AGENT_H_
#define DEEPX_CPP_PTAN_CPP_AGENT_H_

#include <vector>

#include "actions.h"
#include <torch/torch.h>

// Abstract agent interface.
class BaseAgent {
 public:

  // Should create initial empty state for the agent. It will be called for the start of the episode.
  //  Returns anything agent wants to remember.
  virtual torch::Tensor initial_state() {
    return torch::empty({0});
  }

  // Convert observations and states into actions to take.
  // states: list of environment states to process.
  // agent_states: list of states with the same length as observations.
  // Returns tuple of actions, states.
  virtual std::vector<torch::Tensor> operator()(const std::vector<torch::Tensor> &states,
                                                const std::vector<torch::Tensor> &agent_states) = 0;

};

// Wrapper around the model, which provides a copy of it, instead of the trained weights.
class TargetNet {
 public:
  explicit TargetNet(const std::shared_ptr<torch::nn::Module> &model) : model_(model), target_model_(model->clone()) {}

  void sync() {
    target_model_ = model_->clone();
  }

  // Blends parameters of the target net with parameters from the model
  void alpha_sync(double alpha) {
    auto state = model_->named_parameters(/*recurse=*/true);
    auto tgt_state = target_model_->named_parameters(/*recurse=*/true);
    for (auto &item : state) {
      tgt_state[item.key()].add_(1.0 - alpha, alpha);
    }
  }

 private:
  std::shared_ptr<torch::nn::Module> model_;
  std::shared_ptr<torch::nn::Module> target_model_;
};

#endif //DEEPX_CPP_PTAN_CPP_AGENT_H_
