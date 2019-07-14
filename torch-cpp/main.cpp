//
// Created by Saminda Abeyruwan on 2019-07-13.
//

#include <iostream>
#include <string>
#include <torch/torch.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/substitute.h"

ABSL_FLAG(std::string, data_root, "", "Data root");
ABSL_FLAG(int64_t, train_batch_size, 64, "Train batch size");
ABSL_FLAG(int64_t, test_batch_size, 1000, "Test batch size");
ABSL_FLAG(int64_t, number_of_epochs, 5, "Number of epochs");
ABSL_FLAG(int64_t, log_interval, 10, "Log interval");

void testBasicTensorOps() {
  auto mat = torch::rand({3, 3});
  auto mat2 = torch::rand({3, 3});
  auto sum_mat_mat2 = mat + mat2;
  auto sum_mat_mat2_f = torch::add(mat, mat2);
  auto identity = torch::ones({3, 3});
  std::cout << mat << std::endl;
  std::cout << mat * identity << std::endl;
  std::cout << sum_mat_mat2 << std::endl;
  std::cout << sum_mat_mat2_f << std::endl;
}

// Simple DeepNet
class Model : public torch::nn::Module {
 public:
  Model() {
    in_ = register_module("in", torch::nn::Linear(8, 64));
    h_ = register_module("h", torch::nn::Linear(64, 64));
    out_ = register_module("out", torch::nn::Linear(64, 1));
  }

  // the forward operation (how data will flow from layer to layer)
  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(in_->forward(x));
    x = torch::relu(h_->forward(x));
    x = torch::sigmoid(out_->forward(x));
    return x;
  }

 private:
  torch::nn::Linear in_{nullptr}, h_{nullptr}, out_{nullptr};
};

void testModel() {
  Model model;
  auto in = torch::randn({8,});
  auto out = model.forward(in);
  std::cout << in << std::endl;
  std::cout << out << std::endl;
}

// MNIST
class MnistConfig {
 public:
  MnistConfig(const std::string &data_root,
              int64_t train_batch_size,
              int64_t test_batch_size,
              int64_t number_of_epochs,
              int64_t log_interval)
      : data_root_(data_root), train_batch_size_(train_batch_size),
        test_batch_size_(test_batch_size),
        number_of_epochs_(number_of_epochs), log_interval_(log_interval) {}

  std::string DataRoot() const {
    return data_root_;
  }

  int64_t TrainBatchSize() const {
    return train_batch_size_;
  }

  int64_t TestBatchSize() const {
    return test_batch_size_;
  }

  int64_t NumberOfEpochs() const {
    return number_of_epochs_;
  }

  int LogInterval() const {
    return log_interval_;
  }

 private:
  std::string data_root_;
  int64_t train_batch_size_;
  int64_t test_batch_size_;
  int64_t number_of_epochs_;
  int64_t log_interval_;
};

class Net : public torch::nn::Module {
 public:
  Net() : conv1_(torch::nn::Conv2dOptions(1, 10, /*kernel_size*/5)),
          conv2_(torch::nn::Conv2dOptions(10, 20, /*kernel_size*/5)),
          fc1_(320, 50),
          fc2_(50, 10) {
    register_module("conv1", conv1_);
    register_module("conv2", conv2_);
    register_module("conv2_drop", conv2_drop_);
    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(torch::max_pool2d(conv1_->forward(x), 2));
    x = torch::relu(torch::max_pool2d(conv2_drop_->forward(conv2_->forward(x)),
                                      2));
    x = x.view({-1, 320});
    x = torch::relu(fc1_->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*training*/is_training());
    x = fc2_->forward(x);
    return torch::log_softmax(x, /*dim*/1);
  }

 private:
  torch::nn::Conv2d conv1_;
  torch::nn::Conv2d conv2_;
  torch::nn::FeatureDropout conv2_drop_;
  torch::nn::Linear fc1_;
  torch::nn::Linear fc2_;
};

template<typename DataLoader>
void test(
    Net &model,
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
void train(const MnistConfig &config,
           int32_t epoch,
           Net &model,
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

    if (batch_idx++ % config.LogInterval() == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

void testTrainTestMnist() {
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

  Net model;
  model.to(device);

  MnistConfig config(absl::GetFlag(FLAGS_data_root),
                     absl::GetFlag(FLAGS_train_batch_size),
                     absl::GetFlag(FLAGS_test_batch_size),
                     absl::GetFlag(FLAGS_number_of_epochs),
                     absl::GetFlag(FLAGS_log_interval));

  auto train_dataset = torch::data::datasets::MNIST(config.DataRoot())
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), config.TrainBatchSize());

  auto test_dataset = torch::data::datasets::MNIST(
      config.DataRoot(), torch::data::datasets::MNIST::Mode::kTest)
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset),
                                    config.TestBatchSize());

  torch::optim::Adam
      optimizer(model.parameters(), torch::optim::AdamOptions(0.01));

  for (size_t epoch = 1; epoch <= config.NumberOfEpochs(); ++epoch) {
    train(config,
          epoch,
          model,
          device,
          *train_loader,
          optimizer,
          train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
  }

}

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  std::cout << "*** Torch ST ***" << std::endl;
//  testModel();
  testTrainTestMnist();
  std::cout << "*** Torch EN ***" << std::endl;
  return 0;
}
