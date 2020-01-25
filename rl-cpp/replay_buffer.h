//
// Created by saminda on 1/24/20.
//

#ifndef DEEPX_CPP_RL_CPP_REPLAY_BUFFER_H_
#define DEEPX_CPP_RL_CPP_REPLAY_BUFFER_H_

#include <deque>
#include <torch/torch.h>

namespace deepx {

class ReplayBuffer {
 public:
  // Experience of (state, action, reward, next_state, done)
  using Experience = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

  explicit ReplayBuffer(int64_t capacity);

  void push(torch::Tensor state,
            torch::Tensor action,
            torch::Tensor reward,
            torch::Tensor next_state,
            torch::Tensor done);

  std::vector<Experience> sample(int64_t batch_size);

 private:
  int64_t capacity_;
  std::deque<Experience> buffer_;
};

} // namespace deepx


#endif //DEEPX_CPP_RL_CPP_REPLAY_BUFFER_H_
