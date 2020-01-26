//
// Created by saminda on 1/24/20.
//

#include <vector>
#include <algorithm>
#include <iterator>
#include <random>
#include "replay_buffer.h"

namespace deepx {

ReplayBuffer::ReplayBuffer(int64_t capacity) : capacity_(capacity) {}

void ReplayBuffer::push(torch::Tensor state,
                        torch::Tensor action,
                        torch::Tensor reward,
                        torch::Tensor next_state,
                        torch::Tensor done) {
  while (buffer_.size() >= capacity_) {
    buffer_.pop_front();
  }
  buffer_.emplace_back(state, action, reward, next_state, done);
}

std::vector<ReplayBuffer::Experience> ReplayBuffer::sample(int64_t batch_size) {
  std::vector<Experience> b(batch_size);
  std::sample(buffer_.begin(), buffer_.end(), b.begin(), b.size(), std::mt19937{std::random_device{}()});
  return b;
}

int64_t ReplayBuffer::size() const {
  return buffer_.size();
}

} // namespace deepx
