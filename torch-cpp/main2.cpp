//
// Created by saminda on 7/16/19.
//

#include <iostream>

#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <c10/util/Exception.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/substitute.h"

ABSL_FLAG(std::string, data_root, "", "Where to find the CIFAR-10 dataset");
ABSL_FLAG(std::string, output, "", "Where to save pt dataset");

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

struct CIFAR : public Dataset<CIFAR> {
  /// The mode in which the dataset is loaded.
  enum class Mode { kTrain, kTest };

  /// Loads the CIFAR dataset from the `root` path.
  ///
  /// The supplied `root` path should contain the *content* of the unzipped
  /// CIFAR dataset, available from https://www.cs.toronto.edu/~kriz/cifar.html.
  explicit CIFAR(const std::string &root, Mode mode = Mode::kTrain);

  /// Returns the `Example` at the given `index`.
  Example<> get(size_t index) override;

  /// Returns the size of the dataset.
  optional<size_t> size() const override;

  /// Returns true if this is the training subset of CIFAR.
  bool is_train() const noexcept;

  /// Returns all images stacked into a single tensor.
  const Tensor &images() const;

  /// Returns all targets stacked into a single tensor.
  const Tensor &targets() const;

 private:
  bool train_;
  Tensor images_, targets_;
};

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

class BasicBlock : public Module {
 public:
  BasicBlock(uint32_t inplanes,
             uint32_t planes,
             uint32_t stride = 1,
             bool is_downsample = false)
      : stride_(stride), is_downsample_(is_downsample) {
    conv1_ = register_module("conv1", Conv2d(Conv2dOptions(inplanes, planes, 3).stride(stride).padding(1)));
    bn1_ = register_module("bn1", BatchNorm(BatchNormOptions(planes)));
    // ReLU
    conv2_ = register_module("conv2", Conv2d(Conv2dOptions(planes, planes, 3).stride(1).padding(1)));
    bn2_ = register_module("bn2", BatchNorm(BatchNormOptions(planes)));
    if (is_downsample_) {
      downsample_ = register_module("downsample",
                                    Sequential(Conv2d(Conv2dOptions(inplanes, planes, 3).stride(2).padding(1).with_bias(
                                        false))));
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    auto residual = x;
    auto out = torch::relu(bn1_->forward(conv1_->forward(x)));
    out = bn2_->forward(conv2_->forward(out));
    if (downsample_) {
      residual = downsample_->forward(x);
    }
    out += residual;
    return torch::relu(out);
  }

 private:
  uint32_t stride_;
  bool is_downsample_;
  Conv2d conv1_{nullptr};
  BatchNorm bn1_{nullptr};
  // ReLU(inplace=true)
  Conv2d conv2_{nullptr};
  BatchNorm bn2_{nullptr};
  Sequential downsample_{nullptr};
};

class ResNet : public Module {
 public:
  explicit ResNet(uint32_t n = 1) : n_(n) {
    conv1_ = register_module("conv1", Conv2d(Conv2dOptions(3, 16, 3).stride(1).padding(1).with_bias(false)));
    bn1_ = register_module("bn1", BatchNorm(BatchNormOptions(16)));
    // ReLU
    // TODO(saminda): make_layerauto layers =
    layer1_ = MakeLayer(16, 16, 1, "layer1");
    layer2_ = MakeLayer(16, 32, 2, "layer2");
    layer3_ = MakeLayer(32, 64, 2, "layer3");
    fc_ = register_module("fc", Linear(8 * 8 * 64, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(bn1_->forward(conv1_->forward(x))); // 32x32
    x = layer1_->forward(x); // 32x32
    x = layer2_->forward(x); // 16x16
    x = layer3_->forward(x); // 8x8
    x = x.view({-1, 8 * 8 * 64});
    x = fc_->forward(x);
    return torch::log_softmax(x, /*dim*/1);
  }

 private:
  Sequential MakeLayer(uint32_t inplanes, uint32_t planes, uint32_t stride, const std::string &name) {
    auto layers = register_module(name, Sequential());
    layers->push_back(BasicBlock(inplanes, planes, stride, stride != 1));
    // TODO(saminda): n_ blocks
    return layers;
  }

 private:
  uint32_t n_;
  Conv2d conv1_{nullptr};
  BatchNorm bn1_{nullptr};
  // ReLU
  Sequential layer1_{nullptr};
  Sequential layer2_{nullptr};
  Sequential layer3_{nullptr};
  Linear fc_{nullptr};
};

template<typename DataLoader>
void test(
    ResNet &model,
    torch::Device device,
    DataLoader &data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto &batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
        output,
        targets,
        /*weight=*/{},
        Reduction::Sum)
        .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
}

template<typename DataLoader>
void train(int32_t epoch,
           ResNet &model,
           torch::Device device,
           DataLoader &data_loader,
           torch::optim::Optimizer &optimizer,
           size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto &batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % 10 == 0) {
      std::printf(
          "\rTrain Epoch: %d [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

void testTrainTestResNet() {
  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  ResNet model;
  model.to(device);

  auto train_dataset = torch::data::datasets::CIFAR(absl::GetFlag(FLAGS_data_root))
      .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
      .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), 64);

  auto test_dataset = torch::data::datasets::CIFAR(
      absl::GetFlag(FLAGS_data_root), torch::data::datasets::CIFAR::Mode::kTest)
      .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
      .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset),
                                    1024);

  torch::optim::Adam
      optimizer(model.parameters(), torch::optim::AdamOptions(0.01));

  for (size_t epoch = 1; epoch <= 10; ++epoch) {
    train(epoch,
          model,
          device,
          *train_loader,
          optimizer,
          train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
  }

}

void testResNetModel() {
  ResNet res_net;
  std::cout << res_net << std::endl;
  auto x = torch::randn({1, 3, 32, 32});
  auto out = res_net.forward(x);
  std::cout << "res_out: " << out << std::endl;
}

void testCifar() {
  torch::data::datasets::CIFAR cifar(absl::GetFlag(FLAGS_data_root));
  std::cout << cifar.images().sizes() << std::endl;
  std::cout << cifar.images().sizes() << std::endl;
  torch::save(cifar.images(), absl::Substitute("cifar-10-images.pt"));
  torch::save(cifar.targets(), absl::Substitute("cifar-10-targets.pt"));
}

void testNLL() {
  torch::data::datasets::CIFAR cifar(absl::GetFlag(FLAGS_data_root));
  std::cout << cifar.images().sizes() << std::endl;
  std::cout << cifar.images().sizes() << std::endl;

  auto x = cifar.get(0).data;
  auto y = cifar.get(0).target;
  x = torch::stack({x});
  y = torch::stack({y});
  std::cout << x.sizes() << " " << y.sizes() << std::endl;
  ResNet res_net;
  auto pred = res_net.forward(x);
  auto log_softmax = torch::log_softmax(x, 1);
  auto out = torch::nll_loss(pred, y);
  std::cout << "out: " << out << std::endl;
}

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  std::cout << "*** CIFAR ST ***" << std::endl;
//  testCifar();
//  testResNetModel();
  testTrainTestResNet();
//  testNLL();
  std::cout << "*** CIFAR EN ***" << std::endl;
  return 0;
}
