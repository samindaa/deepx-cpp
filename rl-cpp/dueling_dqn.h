//
// Created by saminda on 1/26/20.
//

#ifndef DEEPX_CPP_RL_CPP_DUELING_DQN_H_
#define DEEPX_CPP_RL_CPP_DUELING_DQN_H_

#include "dqn.h"

namespace deepx {

//class DuelingDqn : public DqnModule {
// public:
//  DuelingDqn(int64_t num_inputs, int64_t num_outputs) {
//    feature_ = register_module("feature", torch::nn::Sequential(
//        torch::nn::Linear(num_inputs, 128),
//        torch::nn::Functional(torch::relu)));
//
//    advantage_ = register_module("advantage", torch::nn::Sequential(
//        torch::nn::Linear(128, 128),
//        torch::nn::Functional(torch::relu),
//        torch::nn::Linear(128, num_outputs)));
//
//    value_ = register_module("value", torch::nn::Sequential(
//        torch::nn::Linear(128, 128),
//        torch::nn::Functional(torch::relu),
//        torch::nn::Linear(128, 1)));
//
//  }
//
//  torch::Tensor forward(torch::Tensor x) override {
//    x = feature_->forward(x);
//    auto advantage = advantage_->forward(x);
//    auto value = value_->forward(x);
//    return value + advantage - advantage.mean();
//  }
//
// private:
//  torch::nn::Sequential feature_{nullptr};
//  torch::nn::Sequential advantage_{nullptr};
//  torch::nn::Sequential value_{nullptr};
//};

class DuelingDqn : public DqnModule {
 public:
  DuelingDqn(int64_t num_inputs, int64_t num_outputs) {
    feature_ = register_module("feature", torch::nn::Sequential(
        torch::nn::Linear(num_inputs, 48),
        torch::nn::Functional(torch::relu)));

    advantage_ = register_module("advantage", torch::nn::Sequential(
        torch::nn::Linear(48, 24),
        torch::nn::Functional(torch::relu),
        torch::nn::Linear(24, num_outputs)));

    value_ = register_module("value", torch::nn::Sequential(
        torch::nn::Linear(48, 24),
        torch::nn::Functional(torch::relu),
        torch::nn::Linear(24, 1)));

  }

  torch::Tensor forward(torch::Tensor x) override {
    x = feature_->forward(x);
    auto advantage = advantage_->forward(x);
    auto value = value_->forward(x);
    return value + advantage - advantage.mean();
  }

 private:
  torch::nn::Sequential feature_{nullptr};
  torch::nn::Sequential advantage_{nullptr};
  torch::nn::Sequential value_{nullptr};
};

class DuelingDqnTrainer : public DqnTrainer {
 public:
  DuelingDqnTrainer(std::shared_ptr<Client> client, const EnvConfig &config, int64_t buffer_size = 1000,
                    int64_t batch_size = 32,
                    double epsilon_decay = 500);

 protected:
  torch::Tensor compute_td_loss(int64_t batch_size, float gamma) override;

  void define_models_and_optim() override;
};

} // namespace deepx


#endif //DEEPX_CPP_RL_CPP_DUELING_DQN_H_
