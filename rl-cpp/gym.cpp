//
// Created by saminda on 1/25/20.
//

#include "gym.h"

#include <grpcpp/grpcpp.h>

#include "gym_env.grpc.pb.h"

namespace deepx {

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

std::shared_ptr<Env> Client::make(const std::string &id) {
  std::shared_ptr<Env> env = std::make_shared<Env>();
  return env;
}

std::shared_ptr<Client> Client::create_client(const std::string &addr, const std::string &port) {
  std::shared_ptr<Client> client = std::make_shared<Client>();
  client->stub_ = Gym::NewStub(grpc::CreateChannel(addr + ":" + port, grpc::InsecureChannelCredentials()));
  return client;
}

const EnvConfig Client::Create(const std::string &id, int num_envs) {
  CreateRequest request;
  request.set_id(id);
  request.set_num_envs(num_envs);
  CreateResponse response;
  ClientContext context;

  Status status = stub_->Create(&context, request, &response);

  if (!status.ok()) {
    // TODO(saminda): use status_or
    std::cerr << "gRPC failure" << std::endl;
    exit(-1);
  }
  return response.config();
}

torch::Tensor Client::Reset(const EnvConfig &config) {
  ResetRequest request;
  request.set_env_id(config.env_id());

  ResetResponse response;
  ClientContext context;

  Status status = stub_->Reset(&context, request, &response);

  if (!status.ok()) {
    // TODO(saminda): use status_or
    std::cerr << "gRPC failure" << std::endl;
    exit(-1);
  }

  std::vector<torch::Tensor> states;
  for (const auto &state : response.states()) {
    states.emplace_back(torch::tensor(std::vector<float>{state.obs().begin(), state.obs().end()}));
  }

  return torch::stack(states);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Client::Step(const EnvConfig &config, torch::Tensor action) {
  StepRequest request;
  request.set_env_id(config.env_id());
  // TODO (saminda): generalize this
  if (action.sizes() == at::ArrayRef < int64_t > ({ 1, 1 })) {
    request.add_action(action.item<int>());
  } else {
    std::cerr << "action: " << action << " has dim: " << action.sizes() << " not supported" << std::endl;
    exit(-1);
  }

  StepResponse response;
  ClientContext context;

  Status status = stub_->Step(&context, request, &response);

  if (!status.ok()) {
    // TODO(saminda): use status_or
    std::cerr << "gRPC failure" << std::endl;
    exit(-1);
  }
  std::vector<torch::Tensor> states;
  std::vector<float> rewards;
  std::vector<int> dones;
  for (const auto &step : response.steps()) {
    {
      states.emplace_back(torch::tensor(std::vector<float>{step.state().obs().begin(), step.state().obs().end()}));
      rewards.emplace_back(step.reward());
      dones.emplace_back(step.done());
    }

    return {torch::stack(states), torch::tensor(rewards), torch::tensor(dones)};
  }
}

} // namespace deepx