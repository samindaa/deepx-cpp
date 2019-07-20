//
// Created by saminda on 7/19/19.
//

#include <iostream>
#include <fstream>

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

class MobileNet : public Module {
 public:
  MobileNet() {
    model_ = register_module("model", Sequential());
    conv_bn(model_, 3, 16, 1);
    conv_dw(model_, 16, 16, 1);
    conv_dw(model_, 16, 16, 1);
    conv_dw(model_, 16, 32, 1);

    conv_dw(model_, 32, 32, 2);
    conv_dw(model_, 32, 32, 1);
    conv_dw(model_, 32, 32, 1);
    conv_dw(model_, 32, 64, 1);

    conv_dw(model_, 64, 64, 2);
    conv_dw(model_, 64, 64, 1);
    conv_dw(model_, 64, 64, 1);
    conv_dw(model_, 64, 64, 1);

    fc_ = register_module("fc", Linear(64, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = model_->forward(x);
    x = torch::avg_pool2d(x, 8);
    x = x.view({x.size(0), -1});
    x = fc_->forward(x);
    return torch::log_softmax(x, /*dim*/1);
  }

 private:
  void conv_bn(Sequential &m,
               uint32_t inp,
               uint32_t out,
               uint32_t stride) {
    m->push_back(Conv2d(Conv2dOptions(inp, out, 3).stride(stride).padding(1).with_bias(false)));
    m->push_back(BatchNorm(BatchNormOptions(out)));
    m->push_back(Functional(torch::relu));
  }

  void conv_dw(Sequential &m,
               uint32_t inp,
               uint32_t out,
               uint32_t stride) {
    /*3x3*/
    m->push_back(Conv2d(Conv2dOptions(inp, inp, 3).stride(stride).padding(1).groups(inp).with_bias(false)));
    m->push_back(BatchNorm(BatchNormOptions(inp)));
    m->push_back(Functional(torch::relu));
    /*1x1*/
    m->push_back(Conv2d(Conv2dOptions(inp, out, 1).stride(1).padding(0).with_bias(false)));
    m->push_back(BatchNorm(BatchNormOptions(out)));
    m->push_back(Functional(torch::relu));
  }

 private:
  Sequential model_{nullptr};
  Linear fc_{nullptr};
};

void testMobileNet() {
  MobileNet mobile_net;
  std::cout << mobile_net << std::endl;
  auto x = torch::randn({1, 3, 32, 32});
  auto out = mobile_net.forward(x);
  std::cout << "res_out: " << out << std::endl;
}

template<typename DataLoader>
void test(
    std::shared_ptr<MobileNet> &model,
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
           std::shared_ptr<MobileNet> model,
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

void testTrainTestMobileNet() {
  torch::manual_seed(1);
  std::cout << "Test MobileNet" << std::endl;
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  auto model = std::make_shared<MobileNet>();
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

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  std::cout << "*** Torch ST ***" << std::endl;
//  testMobileNet();
  testTrainTestMobileNet();
  std::cout << "*** Torch EN ***" << std::endl;
  return 0;
}
