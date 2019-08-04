//
// Created by saminda on 8/3/19.
// https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#include <torch/torch.h>

#include "cifar.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/substitute.h"

ABSL_FLAG(std::string, data_root, "", "Where to find the CIFAR-10 dataset");
ABSL_FLAG(int64_t, batch_size, 32, "The batch size for training");
ABSL_FLAG(int64_t, log_interval, 10, "Log interval");
ABSL_FLAG(int64_t, number_of_epochs, 20, "The number of epochs to train");
ABSL_FLAG(int64_t, checkpoint_every, 5, "After how many batches to create a new checkpoint periodically");

using ::torch::nn::Conv2dOptions;
using ::torch::nn::Conv2d;
using ::torch::nn::BatchNorm;
using ::torch::nn::BatchNormOptions;
using ::torch::nn::Module;
using ::torch::nn::Linear;
using ::torch::nn::Sequential;
using ::torch::nn::Functional;
using ::torch::nn::Dropout;
using ::torch::nn::DropoutOptions;

class ConvBNReLU : public Module {
 public:
  ConvBNReLU(uint32_t in_planes,
             uint32_t out_planes,
             uint32_t kernel_size = 3,
             uint32_t stride = 1,
             uint32_t groups = 1) {
    const uint32_t padding = (kernel_size - 1) / 2; // Assumption odd shape kernels
    seq_ = register_module("seq",
                           Sequential(Conv2d(Conv2dOptions(in_planes,
                                                           out_planes,
                                                           kernel_size).stride(stride).padding(padding).groups(groups)
                                                 .with_bias(false)),
                                      BatchNorm(BatchNormOptions(out_planes)),
                                      Functional(torch::relu)));
  }

  torch::Tensor forward(torch::Tensor x) {
    return seq_->forward(x);
  }

 private:
  Sequential seq_{nullptr};
};

class InvertedResidual : public Module {
 public:
  InvertedResidual(uint32_t inp, uint32_t oup, uint32_t stride, uint32_t expansion)
      : use_res_connect_(false) {
    const uint32_t hidden_dim = inp * expansion;
    use_res_connect_ = (stride == 1) && (inp == oup);
    conv_ = register_module("conv", Sequential());
    if (expansion != 1) {
      // pw
      conv_->push_back(ConvBNReLU(inp, hidden_dim, 1 /*kernel_size*/));
    }
    // dw
    conv_->push_back(ConvBNReLU(hidden_dim, hidden_dim, 3 /*kernel_size*/, stride, hidden_dim));
    // pw-linear
    conv_->push_back(Conv2d(Conv2dOptions(hidden_dim, oup, 1/*kernel_size*/).stride(1).padding(0).with_bias(
        false)));
    conv_->push_back(BatchNorm(BatchNormOptions(oup)));
  }

  torch::Tensor forward(torch::Tensor x) {
    if (use_res_connect_) {
      return x + conv_->forward(x);
    }
    return conv_->forward(x);
  }

 private:
  uint32_t use_res_connect_;
  Sequential conv_{nullptr};
};

class MobileNetV2 : public Module {
 public:
  MobileNetV2() {
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> inverted_residual_setting = {{1, 16, 1, 1}, //
                                                                                                 {6, 24, 2,
                                                                                                  1 /*changed stride 2 -> 1 for CIFAR10*/}, //
                                                                                                 {6, 32, 3, 2}, //
                                                                                                 {6, 64, 4, 2}, //
                                                                                                 {6, 96, 3, 1}, //
                                                                                                 {6, 160, 3, 2}, //
                                                                                                 {6, 320, 1, 1}
    }; // (t, c, n, s)

    uint32_t input_channel = 32;
    const uint32_t last_channel = 1280;
    const uint32_t num_classes = 10;

    features_ = register_module("features", Sequential());
    features_->push_back(ConvBNReLU(3, input_channel, 3, 2));
    // Building inverted residual blocks
    uint32_t t, c, n, s;
    for (const auto &tcns : inverted_residual_setting) {
      std::tie(t, c, n, s) = tcns;
      for (uint32_t i = 0; i < n; ++i) {
        const uint32_t stride = (i == 0) ? s : 1;
        features_->push_back(InvertedResidual(input_channel, c, stride, t));
        input_channel = c;
      }
    }
    // Building last several layers
    features_->push_back(ConvBNReLU(input_channel, last_channel, 1));
    classifier_ =
        register_module("classifier", Sequential(Dropout(DropoutOptions(0.2)), Linear(last_channel, num_classes)));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = features_->forward(x);
    x = x.mean({2, 3});
    x = classifier_->forward(x);
    return torch::log_softmax(x, /*dim*/1);
  }

 private:
  Sequential features_{nullptr};
  Sequential classifier_{nullptr};
};

void testMobileNetV2() {
  MobileNetV2 mobilenetv2;
  std::cout << mobilenetv2 << std::endl;
  auto x = torch::randn({1, 3, 32, 32});
  auto out = mobilenetv2.forward(x);
  std::cout << "res_out: " << out << std::endl;
}

template<typename DataLoader>
void test(
    std::shared_ptr<MobileNetV2> &model,
    torch::Device device,
    DataLoader &data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model->eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto &batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model->forward(data);
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
           std::shared_ptr<MobileNetV2> model,
           torch::Device device,
           DataLoader &data_loader,
           torch::optim::Optimizer &optimizer,
           size_t dataset_size) {
  model->train();
  size_t batch_idx = 0;
  for (auto &batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model->forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % absl::GetFlag(FLAGS_log_interval) == 0) {
      std::printf(
          "\rTrain Epoch: %d [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

void testTrainTestMobileNetV2() {
  torch::manual_seed(1);
  std::cout << "Test MobileNetV2" << std::endl;
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  auto model = std::make_shared<MobileNetV2>();
  model->to(device);

  auto train_dataset = torch::data::datasets::CIFAR(absl::GetFlag(FLAGS_data_root))
      .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
      .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), absl::GetFlag(FLAGS_batch_size));

  auto test_dataset = torch::data::datasets::CIFAR(
      absl::GetFlag(FLAGS_data_root), torch::data::datasets::CIFAR::Mode::kTest)
      .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
      .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset),
                                    1024);

  torch::optim::Adam
      optimizer(model->parameters(), torch::optim::AdamOptions(0.01));

  for (size_t epoch = 1; epoch <= absl::GetFlag(FLAGS_number_of_epochs); ++epoch) {
    train(epoch,
          model,
          device,
          *train_loader,
          optimizer,
          train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
    if (epoch % absl::GetFlag(FLAGS_checkpoint_every) == 0) {
      // Checkpoint the model and optimizer state.
      std::ofstream archive("mobilenet-checkpoint.pt");
      torch::save(model, archive);
      torch::save(optimizer, "mobilenet-optimizer-checkpoint.pt");
    }
  }
}

void testModelLoad() {
  std::cout << "Test MobileNet Load" << std::endl;
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);
  auto model = std::make_shared<MobileNetV2>();
  torch::load(model, "mobilenet-checkpoint.pt");
  auto test_dataset = torch::data::datasets::CIFAR(
      absl::GetFlag(FLAGS_data_root), torch::data::datasets::CIFAR::Mode::kTest)
      .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465}, {0.2023, 0.1994, 0.2010}))
      .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset),
                                    1024);
  test(model, device, *test_loader, test_dataset_size);

}

//void testTensor() {
//  auto p = [](torch::Tensor x) {
//    std::cout << x.sizes() << std::endl;
//    std::cout << x << std::endl;
//    std::cout << "---" << std::endl;
//  };
//  auto x = torch::tensor({1, 2, 3}, torch::kFloat32).unsqueeze(1).unsqueeze(2);
//  p(x);
//  auto x1 = torch::randn({3, 3, 3});
//  p(x1);
//  p(x1 + torch::tensor({100, 200, 300}, torch::kFloat32).unsqueeze(1).unsqueeze(2));
//}

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  std::cout << "*** Torch ST ***" << std::endl;
//  testMobileNetV2();
  testTrainTestMobileNetV2();
  std::cout << "*** Torch EN ***" << std::endl;
  return 0;
}
