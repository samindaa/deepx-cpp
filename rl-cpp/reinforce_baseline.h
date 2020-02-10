//
// Created by saminda on 2/3/20.
//

#ifndef DEEPX_CPP_RL_CPP_REINFORCE_BASELINE_H_
#define DEEPX_CPP_RL_CPP_REINFORCE_BASELINE_H_

#include "gym.h"

#include <random>
#include <vector>
#include <torch/torch.h>

namespace deepx {

class PolicyNetwork : public torch::nn::Module {
 public:
  PolicyNetwork(int64_t n_state, int64_t n_action, int64_t n_hidden) {
    layers_ = register_module("layers", torch::nn::Sequential(
        torch::nn::Linear(n_state, n_hidden),
        torch::nn::Dropout(torch::nn::DropoutOptions(/*rate=*/0.6)),
        torch::nn::Functional(torch::relu),
        torch::nn::Linear(n_hidden, n_action),
        torch::nn::Functional(torch::sigmoid)));
  }

  torch::Tensor forward(torch::Tensor x) {
    return layers_->forward(x);
  }

  std::tuple<torch::Tensor, torch::Tensor> get_action(torch::Tensor state) {
    torch::Tensor probs = forward(state);
    torch::Tensor action = torch::multinomial(probs, /*num_samples=*/1);
    // Assuming state is (1, n)
    torch::Tensor log_prob = torch::log(probs[0][action.item<int64_t>()]);
    return {action, log_prob};
  }

 private:
  torch::nn::Sequential layers_{nullptr};
};


class ReinforceBaselineTrainer {
 public:
  ReinforceBaselineTrainer(std::shared_ptr<Client> client,
                           const EnvConfig &config);

  void train(int64_t num_frames);

  void test(bool render = false);

  virtual void define_models_and_optim();

 protected:
  virtual torch::Tensor compute_td_loss(int64_t batch_size, float gamma);

  torch::Tensor get_state_tensor(const State &state) const;

 protected:
  float running_reward_;
  float gamma_ = 0.99;
  std::shared_ptr<Client> client_;
  EnvConfig config_;
  std::shared_ptr<PolicyNetwork> policy_net_;
  std::shared_ptr<torch::optim::Adam> policy_opt_;
  std::mt19937 rand_generator_;
  torch::Device device_;
  std::vector<torch::Tensor> saved_log_probs_;
  std::vector<float> rewards_;
};

} // namespace deepx

#endif //DEEPX_CPP_RL_CPP_REINFORCE_BASELINE_H_
