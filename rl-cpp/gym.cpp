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

EnvConfig Client::Create(const std::string& id) {
  CreateRequest request;
  request.set_id(id);
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

State Client::Reset(const EnvConfig& config) {
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
  return response.state();
}

deepx::Step Client::Step(const EnvConfig& config, torch::Tensor action, bool render) {
  StepRequest request;
  request.set_env_id(config.env_id());
  request.set_render(render);
  auto cpu_action = action.cpu();
  // TODO(saminda): assuming single invocation
  for (int64_t i = 0; i < cpu_action.numel(); ++i) {
    request.add_action(cpu_action[i].item<int>());
  }

  StepResponse response;
  ClientContext context;

  Status status = stub_->Step(&context, request, &response);

  if (!status.ok()) {
    // TODO(saminda): use status_or
    std::cerr << "gRPC failure" << std::endl;
    exit(-1);
  }
  return response.step();
}

} // namespace deepx