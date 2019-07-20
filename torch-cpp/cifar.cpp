//
// Created by saminda on 7/19/19.
//
#include "cifar.h"

#include <cassert>
#include <iostream>

#include <torch/torch.h>

#include "absl/strings/substitute.h"

using ::torch::nn::Conv2dOptions;
using ::torch::nn::Conv2d;
using ::torch::nn::BatchNorm;
using ::torch::nn::BatchNormOptions;
using ::torch::nn::Module;
using ::torch::nn::Linear;
using ::torch::nn::Sequential;

namespace torch {
namespace data {
namespace datasets {
namespace {
constexpr uint32_t kCIFARSize = 32;
constexpr uint32_t kCIFAR10BatchSize = 10000;
constexpr uint32_t kCIFAR10TrainBatches = 5;

constexpr const char *kTrainFilenamePrefix = "data_batch";
constexpr const char *kTestFilename = "test_batch.bin";

std::string join_paths(std::string head, const std::string &tail) {
  if (head.back() != '/') {
    head.push_back('/');
  }
  head += tail;
  return head;
}

void read_file(std::ifstream &file, std::vector<Tensor> &x, std::vector<Tensor> &y) {
  for (uint32_t i = 0; i < kCIFAR10BatchSize; ++i) {
    auto y_tensor = torch::empty(1, torch::kByte);
    file.read(reinterpret_cast<char *>(y_tensor.data_ptr()), y_tensor.numel());
    auto x_tensor = torch::empty({1, 3, kCIFARSize, kCIFARSize}, torch::kByte);
    file.read(reinterpret_cast<char *>(x_tensor.data_ptr()), x_tensor.numel());
    x.emplace_back(x_tensor);
    y.emplace_back(y_tensor);
  }
}

std::pair<Tensor, Tensor> read_images_targets(const std::string &root, bool train) {
  std::vector<Tensor> x, y;
  if (train) {
    for (uint32_t i = 1; i <= kCIFAR10TrainBatches; ++i) {
      const auto path = join_paths(root, absl::Substitute("$0_$1.bin", kTrainFilenamePrefix, i));
      std::ifstream file(path, std::ios::binary);
      if (!file.is_open()) {
        std::cerr << absl::Substitute("Path: $0 is not found.", path) << std::endl;
        exit(-1);
      }
      read_file(file, x, y);
    }
  } else {
    const auto path = join_paths(root, kTestFilename);
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << absl::Substitute("Path: $0 is not found.", path) << std::endl;
      exit(-1);
    }
    read_file(file, x, y);
  }
  return {torch::cat(x, 0).to(torch::kFloat32).div_(255), torch::cat(y, 0).to(torch::kInt64)};
}

} // namespace

CIFAR::CIFAR(const std::string &root, Mode mode) : train_(mode == Mode::kTrain) {
  std::tie(images_, targets_) = read_images_targets(root, train_);
}

Example<> CIFAR::get(size_t index) {
  return {images_[index], targets_[index]};
}

optional<size_t> CIFAR::size() const {
  return images_.size(0);
}

bool CIFAR::is_train() const noexcept {
  return train_;
}

const Tensor &CIFAR::images() const {
  return images_;
}

const Tensor &CIFAR::targets() const {
  return targets_;
}

} // namespace datasets
} // namespace data
} // namespace torch
