//
// Created by saminda on 1/24/20.
//

#ifndef DEEPX_CPP_RL_CPP_GYM_H_
#define DEEPX_CPP_RL_CPP_GYM_H_

#include <memory>
#include <torch/torch.h>
#include "gym_env.pb.h"
#include "gym_env.grpc.pb.h"

namespace deepx {

// Act as the remote proxy
class Env {
 public:
  // TODO(saminda):
};

class Client {
 public:
  std::shared_ptr<Env> make(const std::string &id);
  static std::shared_ptr<Client> create_client(const std::string &addr, const std::string& port);

  EnvConfig Create(const std::string& id);
  State Reset(const EnvConfig& config);
  deepx::Step Step(const EnvConfig& config, torch::Tensor action, bool render = false);

  Client() = default;

 private:
  std::shared_ptr<Gym::Stub> stub_;
};

} // namespace deepx

#endif //DEEPX_CPP_RL_CPP_GYM_H_
