//
// Created by saminda on 1/24/20.
//

#include <iostream>
#include "gym.h"

void TestClient() {
  auto client = deepx::Client::create_client("localhost", "50051");
  auto config = client->Create("CartPole-v1");
  std::cout << config.DebugString() << std::endl;
  auto state = client->Reset(config);
  std::cout << state.DebugString() << std::endl;
  while (true) {
    auto step = client->Step(config, torch::zeros({1}));
    std::cout <<  step.DebugString() << std::endl;
    if (step.done()) {
      break;
    }
  }
}

// TODO(saminda): implement the rest of the stuff
int main(int argc, char** argv) {
  std::cout << "*** start ***" << std::endl;
  TestClient();
  std::cout << "** end    ***" << std::endl;
  return 0;
}