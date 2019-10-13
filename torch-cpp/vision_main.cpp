//
// Created by saminda on 9/6/19.
//
#include "squeezenet.h"
#include "mobilenet.h"

#include "matplotlibcpp.h"
#include "cifar.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/substitute.h"

ABSL_FLAG(std::string,
          data_root,
          "/home/saminda/Data/cifar-10/cifar-10-batches-bin",
          "Where to find the CIFAR-10 dataset");
ABSL_FLAG(int64_t, batch_size, 32, "The batch size for training");
ABSL_FLAG(int64_t, log_interval, 10, "Log interval");
ABSL_FLAG(int64_t, number_of_epochs, 20, "The number of epochs to train");
ABSL_FLAG(int64_t, checkpoint_every, 5, "After how many batches to create a new checkpoint periodically");

namespace {
using ::torch::nn::Module;
using ::torch::nn::Sequential;
using ::vision::models::SqueezeNet1_1;
namespace plt = matplotlibcpp;

class SqueezeNet : public Module {
 public:
  explicit SqueezeNet() {
    seq_ = register_module("seq", Sequential(SqueezeNet1_1(/*num_classes=*/10)));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = seq_->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

 private:
  Sequential seq_{nullptr};
};

class MobileNetV2 : public Module {
 public:
  explicit MobileNetV2() {
    std::vector<std::vector<int64_t>> inverted_residual_settings = {
        // t, c, n, s
        {1, 16, 1, 1},
        {6, 24, 2, 1 /*changed stride 2 -> 1 for CIFAR10*/},
        {6, 32, 3, 2},
        {6, 64, 4, 2},
        {6, 96, 3, 1},
        {6, 160, 3, 2},
        {6, 320, 1, 1},
    };
    seq_ = register_module("seq",
                           Sequential(::vision::models::MobileNetV2(/*num_classes=*/10, /*width_mult=*/
                                                                                    1.0,
                                                                                    inverted_residual_settings)));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = seq_->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

 private:
  Sequential seq_{nullptr};
};

void TestSqueezeNet1_1() {
  auto net = std::make_shared<SqueezeNet>();
  auto x = torch::randn(/*size=*/{5, 3, 32, 32});
  std::cout << x.sizes() << std::endl;
  auto y = net->forward(x);
  std::cout << y.sizes() << std::endl;
}

void TestMobileNetV2() {
  auto net = std::make_shared<MobileNetV2>();
  auto x = torch::randn(/*size=*/{5, 3, 32, 32});
  std::cout << x.sizes() << std::endl;
  auto y = net->forward(x);
  std::cout << y.sizes() << std::endl;
}

void TestCifar() {
  const size_t num_rows = 5;
  const size_t num_cols = 5;
  auto
      dataset = torch::data::datasets::CIFAR(absl::GetFlag(FLAGS_data_root), torch::data::datasets::CIFAR::Mode::kTest);
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      auto example = dataset.get(i * num_cols + j);
      auto data = example.data;
      data = torch::cat(/*tensors=*/{data[0].view({32, 32, 1}), data[1].view({32, 32, 1}),
                                     data[2].view({32, 32, 1})}, /*dim=*/2);
//      plt::title(absl::Substitute("$0", example.target.item<int32_t>()));
      plt::subplot(num_rows, num_cols, i * num_cols + j + 1);
      plt::imshow(data.data_ptr<float>(), data.size(0), data.size(1), data.size(2));
    }
  }
  plt::show();
}

template<typename Model, typename DataLoader>
void Train(int32_t epoch,
           std::shared_ptr<Model> model,
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

template<typename Model, typename DataLoader>
void Test(
    std::shared_ptr<Model> model,
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

void Runner() {
  torch::manual_seed(1);
  std::cout << "Test Vision" << std::endl;
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
      .map(torch::data::transforms::Normalize<>(/*mean=*/{0.4914, 0.4822, 0.4465}, /*stddev=*/{0.2023, 0.1994, 0.2010}))
      .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), absl::GetFlag(FLAGS_batch_size));

  auto test_dataset = torch::data::datasets::CIFAR(
      absl::GetFlag(FLAGS_data_root), torch::data::datasets::CIFAR::Mode::kTest)
      .map(torch::data::transforms::Normalize<>(/*mean=*/{0.4914, 0.4822, 0.4465}, /*stddev=*/{0.2023, 0.1994, 0.2010}))
      .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset),
          /*options=*/1024);

  torch::optim::Adam
      optimizer(model->parameters(), torch::optim::AdamOptions(/*learning_rate=*/0.01));

  for (size_t epoch = 1; epoch <= absl::GetFlag(FLAGS_number_of_epochs); ++epoch) {
    Train(epoch,
          model,
          device,
          *train_loader,
          optimizer,
          train_dataset_size);
    Test(model, device, *test_loader, test_dataset_size);
    if (epoch % absl::GetFlag(FLAGS_checkpoint_every) == 0) {
      // Checkpoint the model and optimizer state.
      std::ofstream archive("squeezenet-checkpoint.pt");
      torch::save(model, archive);
      torch::save(optimizer, "squeezenet-optimizer-checkpoint.pt");
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
//  TestSqueezeNet1_1();
//  TestMobileNetV2();
  Runner();
//  TestCifar();
  return 0;
}