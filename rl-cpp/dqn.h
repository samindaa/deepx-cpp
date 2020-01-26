//
// Created by saminda on 1/24/20.
//

#ifndef DEEPX_CPP_RL_CPP_DQN_H_
#define DEEPX_CPP_RL_CPP_DQN_H_

#include "replay_buffer.h"
#include "gym.h"

#include <random>
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

class DqnTrainer {
 public:
  DqnTrainer(std::shared_ptr<Client> client, const EnvConfig& config, int64_t buffer_size);

  void train(int64_t num_frames);

  void test(bool render = false);

 protected:
  virtual double epsilon_by_frame(int64_t frame_id);

  virtual torch::Tensor compute_td_loss(int64_t batch_size, float gamma);

  torch::Tensor get_state_tensor(const State& state) const;

 private:
  int64_t batch_size_ = 32;
  double epsilon_start_ = 1.0;
  double epsilon_final_ = 0.01;
  double epsilon_decay_ = 500;
  float gamma_ = 0.99;
  std::shared_ptr<Client> client_;
  EnvConfig config_;
  ReplayBuffer buffer_;
  std::shared_ptr<Dqn> current_model_;
  std::shared_ptr<Dqn> target_model_;
  torch::optim::Adam opt_;
  std::mt19937 rand_generator_;
  torch::Device device_;
};


} // namespace deepx


#endif //DEEPX_CPP_RL_CPP_DQN_H_
