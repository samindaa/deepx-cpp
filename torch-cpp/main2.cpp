//
// Created by saminda on 7/16/19.
//

#include <cassert>
#include <iostream>

#include <torch/torch.h>

#include "cifar.h"
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

class BasicBlock : public Module {
 public:
  BasicBlock(uint32_t in_planes,
             uint32_t planes,
             uint32_t stride = 1,
             bool is_downsample = false)
      : stride_(stride), is_downsample_(is_downsample) {
    conv1_ = register_module("conv1",
                             Conv2d(Conv2dOptions(in_planes, planes, 3).stride(stride).padding(1).with_bias(false)));
    conv2_ = register_module("conv2", Conv2d(Conv2dOptions(planes, planes, 3).stride(1).padding(1).with_bias(false)));

    bn1_ = register_module("bn1", BatchNorm(BatchNormOptions(planes)));
    bn2_ = register_module("bn2", BatchNorm(BatchNormOptions(planes)));

    if (is_downsample_) {
      downsample_ = register_module("downsample",
                                    Sequential(Conv2d(Conv2dOptions(in_planes,
                                                                    planes,
                                                                    3).stride(2).padding(1).with_bias(
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
  explicit ResNet(uint32_t n) {
    assert((n - 2) % 6 == 0); // Depth should be 6n+2
    n_ = (n - 2) / 6;
    conv_ = register_module("conv1", Conv2d(Conv2dOptions(3, 16, 3).stride(1).padding(1).with_bias(false)));
    bn_ = register_module("bn1", BatchNorm(BatchNormOptions(16)));
    // ReLU
    // TODO(saminda): make_layerauto layers =
    layer1_ = MakeLayer(16, 16, 1, "layer1");
    layer2_ = MakeLayer(16, 32, 2, "layer2");
    layer3_ = MakeLayer(32, 64, 2, "layer3");
    fc_ = register_module("fc", Linear(64, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(bn_->forward(conv_->forward(x))); // 32x32
    x = layer1_->forward(x); // 32x32
    x = layer2_->forward(x); // 16x16
    x = layer3_->forward(x); // 8x8
    x = torch::avg_pool2d(x, 8, 1); // 1x1
    x = x.view({x.size(0), -1});
    x = fc_->forward(x);
    return torch::log_softmax(x, /*dim*/1);
  }

 private:
  Sequential MakeLayer(uint32_t in_planes, uint32_t planes, uint32_t stride, const std::string &name) {
    auto layers = register_module(name, Sequential());
    layers->push_back(BasicBlock(in_planes, planes, stride, stride != 1));
    for (uint32_t i = 1; i < n_; ++i) {
      layers->push_back(BasicBlock(planes, planes));
    }
    return layers;
  }

 private:
  uint32_t n_;
  Conv2d conv_{nullptr};
  BatchNorm bn_{nullptr};
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

  ResNet model(20);
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

  for (size_t epoch = 1; epoch <= 50; ++epoch) {
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
  ResNet res_net(20);
  std::cout << res_net << std::endl;
  auto x = torch::randn({1, 3, 32, 32});
  auto out = res_net.forward(x);
  std::cout << "res_out: " << out << std::endl;
//  std::cout << res_net << std::endl;
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
  ResNet res_net(20);
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
