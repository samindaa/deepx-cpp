//
// Created by saminda on 12/31/19.
//

#include <iostream>

#include "actions.h"

void TestActions() {
  auto scores = torch::tensor({{1.0, 2.0, 3.0}, {2.0, 1.0, 1.0}});
  ArgmaxActionSelector argmax_action_selector;
  auto actions = argmax_action_selector(scores);
  std::cout << actions << std::endl;

  EpsilonGreedyActionSelector epsilon_greedy_action_selector(/*epsilon=*/0.8, /*device=*/torch::kCUDA);
  actions = epsilon_greedy_action_selector(scores.to(torch::kCUDA));
  std::cout << actions << std::endl;
}

void TestRandom() {
  auto rand = torch::rand({5, 1});
  std::cout << rand << std::endl;
  auto rand1 = rand.lt(0.01).nonzero_numpy()[0];
  std::cout << rand1 << std::endl;
}

class Net : public torch::nn::Module {
 public:
  Net() : conv1_(torch::nn::Conv2dOptions(1, 10, /*kernel_size*/5)),
          conv2_(torch::nn::Conv2dOptions(10, 20, /*kernel_size*/5)),
          fc1_(320, 50),
          fc2_(50, 10) {
    register_module("conv1", conv1_);
    register_module("conv2", conv2_);
    register_module("conv2_drop", conv2_drop_);
    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(torch::max_pool2d(conv1_->forward(x), 2));
    x = torch::relu(torch::max_pool2d(conv2_drop_->forward(conv2_->forward(x)),
                                      2));
    x = x.view({-1, 320});
    x = torch::relu(fc1_->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*training*/is_training());
    x = fc2_->forward(x);
    return torch::log_softmax(x, /*dim*/1);
  }

 private:
  torch::nn::Conv2d conv1_;
  torch::nn::Conv2d conv2_;
  torch::nn::FeatureDropout conv2_drop_;
  torch::nn::Linear fc1_;
  torch::nn::Linear fc2_;
};

void TestNamedParams() {
  auto net = std::make_shared<Net>();
  for (const auto& item : net->named_parameters(true)) {
    std::cout << "name: " << item.key() << std::endl;
    std::cout << "param.shape: " << item.value().sizes() << std::endl;
    std::cout << "param.requires_grad: " << item.value().requires_grad() << std::endl;
  }
}

int main(int argc, char **argv) {
  std::cout << "*** start ***" << std::endl;
//  TestActions();
//  TestRandom();
  TestNamedParams();
  std::cout << "*** end   ***" << std::endl;
  return 0;
}
