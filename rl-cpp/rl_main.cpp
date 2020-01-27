//
// Created by saminda on 1/24/20.
//

#include <iostream>
#include "common.h"
#include "gym.h"
#include "dqn.h"
#include "dueling_dqn.h"

void TestClient() {
  auto client = deepx::Client::create_client("localhost", "50051");
  auto config = client->Create("CartPole-v1");
  std::cout << config.DebugString() << std::endl;
  auto state = client->Reset(config);
  std::cout << state.DebugString() << std::endl;
  while (true) {
    auto step = client->Step(config, torch::zeros({1}));
    std::cout << step.DebugString() << std::endl;
    if (step.done()) {
      break;
    }
  }
}

class TestModule : public torch::nn::Module {
 public:
  explicit TestModule(bool override = false) {
    h1 = register_module("h1", torch::nn::Linear(3, 4));
    h2 = register_module("h2", torch::nn::Linear(4, 2));

    if (override) {
      for (auto &module : modules(/*include_self=*/false)) {
        if (auto m = dynamic_cast<torch::nn::LinearImpl *>(module.get())) {
          torch::nn::init::ones_(m->weight);
          torch::nn::init::ones_(m->bias);
        }
      }
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    return h2->forward(h1->forward(x));
  }

 private:
  torch::nn::Linear h1{nullptr};
  torch::nn::Linear h2{nullptr};
};

void TestLoadFromStateDict() {
  std::shared_ptr<TestModule> src = std::make_shared<TestModule>(true), tgt = std::make_shared<TestModule>();

  auto print_params = [](std::shared_ptr<TestModule> mod) {
    for (auto &module : mod->modules()) {
      if (auto m = dynamic_cast<torch::nn::LinearImpl *>(module.get())) {
        std::cout << "name:   " << m->name() << std::endl;
        std::cout << "weight: " << m->weight << std::endl;
        std::cout << "bias:   " << m->bias << std::endl;
      }
    }
  };

  print_params(src);
  print_params(tgt);

  deepx::load_from_state_dict(tgt, src);

  print_params(src);
  print_params(tgt);
}

void TestDqn() {
  auto client = deepx::Client::create_client("localhost", "50051");
  auto config = client->Create("CartPole-v1");
  std::shared_ptr<deepx::DqnTrainer>
      dqn_trainer = std::make_shared<deepx::DqnTrainer>(client, config, /*buffer_size=*/10000);
  dqn_trainer->define_models_and_optim();
  dqn_trainer->train(/*num_frames=*/10000);
  dqn_trainer->test(true);
}

void TestDuelingDqn() {
  auto client = deepx::Client::create_client("localhost", "50051");
  auto config = client->Create("CartPole-v1");
  std::shared_ptr<deepx::DqnTrainer>
      dqn_trainer = std::make_shared<deepx::DuelingDqnTrainer>(client, config, /*buffer_size=*/10000);
  dqn_trainer->define_models_and_optim();
  dqn_trainer->train(/*num_frames=*/10000);
  dqn_trainer->test(true);
}

// TODO(saminda): implement the rest of the stuff
int main(int argc, char **argv) {
  std::cout << "*** start ***" << std::endl;
//  TestClient();
//  TestLoadFromStateDict();
//  TestDqn();
  TestDuelingDqn();
  std::cout << "** end    ***" << std::endl;
  return 0;
}