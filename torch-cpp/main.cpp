//
// Created by Saminda Abeyruwan on 2019-07-13.
//

#include <string>
#include <vector>
#include <sstream>
#include <iostream>

#include <torch/torch.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/substitute.h"

ABSL_FLAG(std::string, data_root, "", "Where to find the MNIST dataset");
ABSL_FLAG(int64_t, train_batch_size, 64, "The batch size for training");
ABSL_FLAG(int64_t, test_batch_size, 1000, "Test batch size");
ABSL_FLAG(int64_t, number_of_epochs, 30, "The number of epochs to train");
ABSL_FLAG(int64_t, batch_size, 64, "The batch size for training");
ABSL_FLAG(int64_t, log_interval, 10, "Log interval");
ABSL_FLAG(int64_t, noise_size, 100, "The size of the noise vector fed to the generator");
ABSL_FLAG(int64_t, checkpoint_every, 200, "After how many batches to create a new checkpoint periodically");
ABSL_FLAG(int64_t, number_of_samples_per_checkpoint, 10, "How many images to sample at every checkpoint");
ABSL_FLAG(bool,
          restore_from_checkpoint,
          false,
          "Set to 'true' to restore models and optimizers from previously saved checkpoints");

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

// Regression
torch::Tensor make_feature(torch::Tensor x, int64_t poly_degree) {
  x = x.unsqueeze(1);
  std::vector<torch::Tensor> xs;
  for (int64_t i = 0; i < poly_degree; ++i) {
    xs.emplace_back(x.pow(i + 1));
  }
  return torch::cat(xs, 1);
}

torch::Tensor f(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
  return x.mm(w) + b.item();
}

std::pair<torch::Tensor, torch::Tensor> get_batch(torch::Tensor w,
                                                  torch::Tensor b,
                                                  int64_t poly_degree,
                                                  int64_t batch_size = 32) {
  auto random = torch::randn({batch_size});
  auto x = make_feature(random, poly_degree);
  auto y = f(x, w, b);
  return {x, y};
}

std::string poly_desc(torch::Tensor w, torch::Tensor b) {
  auto size = w.size(0);
  std::ostringstream stream;
  stream << "y = ";
  for (int64_t i = 0; i < size; ++i) {
    stream << w[i].item<float>() << " x^" << size - i << " ";
  }
  stream << "+ " << b[0].item<float>();
  return stream.str();
}

void testPoly() {
  const int64_t poly_degree = 4;
  auto w_target = torch::randn({poly_degree, 1}) * 5;
  auto b_target = torch::randn({1}) * 5;

  // Define the model and optimizer
  auto fc = torch::nn::Linear(w_target.size(0), 1);
  torch::optim::SGD optim(fc->parameters(), 0.1);

  float loss = 0;
  int64_t batch_idx = 0;

  while (++batch_idx) {
    torch::Tensor batch_x, batch_y;
    std::tie(batch_x, batch_y) = get_batch(w_target, b_target, poly_degree);

    // Reset gradients
    optim.zero_grad();

    // Forward pass
    auto output = torch::smooth_l1_loss(fc(batch_x), batch_y);
    loss = output.item<float>();

    // Backward pass
    output.backward();

    // Apply gradients
    optim.step();

    if (loss < 1e-3f) {
      break;
    }
  }

  std::cout << absl::Substitute("Loss: $0 after $1 batches", loss, batch_idx) << std::endl;
  std::cout << absl::Substitute("==> Learned function:\t $0", poly_desc(fc->weight.view({-1}), fc->bias)) << std::endl;
  std::cout << absl::Substitute("==> Actual function:\t $0", poly_desc(w_target.view({-1}), b_target)) << std::endl;
}

void testDcgan() {
  torch::manual_seed(1);

  // Create the device we pass around based on whether CUDA is available.
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU" << std::endl;
    device = torch::Device(torch::kCUDA);
  }

  torch::nn::Sequential generator(
      // Layer 1
      torch::nn::Conv2d(torch::nn::Conv2dOptions(absl::GetFlag(FLAGS_noise_size), 256, 4)
                            .with_bias(false)
                            .transposed(true)),
      torch::nn::BatchNorm(256),
      torch::nn::Functional(torch::relu),
      // Layer 2
      torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 128, 3)
                            .stride(2)
                            .padding(1)
                            .with_bias(false)
                            .transposed(true)),
      torch::nn::BatchNorm(128),
      torch::nn::Functional(torch::relu),
      // Layer 3
      torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 64, 4)
                            .stride(2)
                            .padding(1)
                            .with_bias(false)
                            .transposed(true)),
      torch::nn::BatchNorm(64),
      torch::nn::Functional(torch::relu),
      // Layer 4
      torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 1, 4)
                            .stride(2)
                            .padding(1)
                            .with_bias(false)
                            .transposed(true)),
      torch::nn::Functional(torch::tanh)
  );
  generator->to(device);

  torch::nn::Sequential discriminator(
      // Layer 1
      torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).with_bias(false)),
      torch::nn::Functional(torch::leaky_relu, 0.2),
      // Layer 2
      torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).with_bias(false)),
      torch::nn::BatchNorm(128),
      torch::nn::Functional(torch::leaky_relu, 0.2),
      // Layer 3
      torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).with_bias(false)),
      torch::nn::BatchNorm(256),
      torch::nn::Functional(torch::leaky_relu, 0.2),
      // Layer 4
      torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).with_bias(false)),
      torch::nn::Functional(torch::sigmoid));
  discriminator->to(device);


  // Assume the MNIST dataset is available under `kDataFolder`;
  auto dataset = torch::data::datasets::MNIST(absl::GetFlag(FLAGS_data_root))
      .map(torch::data::transforms::Normalize<>(0.5, 0.5))
      .map(torch::data::transforms::Stack<>());
  const int64_t batches_per_epoch =
      std::ceil(dataset.size().value() / static_cast<double>(absl::GetFlag(FLAGS_batch_size)));

  auto data_loader = torch::data::make_data_loader(
      std::move(dataset),
      torch::data::DataLoaderOptions().batch_size(absl::GetFlag(FLAGS_batch_size)).workers(2));

  torch::optim::Adam generator_optimizer(
      generator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
  torch::optim::Adam discriminator_optimizer(
      discriminator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));

  if (absl::GetFlag(FLAGS_restore_from_checkpoint)) {
    torch::load(generator, "generator-checkpoint.pt");
    torch::load(generator_optimizer, "generator-optimizer-checkpoint.pt");
    torch::load(discriminator, "discriminator-checkpoint.pt");
    torch::load(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
  }

  int64_t checkpoint_counter = 1;
  for (int64_t epoch = 1; epoch <= absl::GetFlag(FLAGS_number_of_epochs); ++epoch) {
    int64_t batch_index = 0;
    for (torch::data::Example<> &batch : *data_loader) {
      // Train discriminator with real images.
      discriminator->zero_grad();
      torch::Tensor real_images = batch.data.to(device);
      torch::Tensor real_labels =
          torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
      torch::Tensor real_output = discriminator->forward(real_images);
      torch::Tensor d_loss_real =
          torch::binary_cross_entropy(real_output, real_labels);
      d_loss_real.backward();

      // Train discriminator with fake images.
      torch::Tensor noise =
          torch::randn({batch.data.size(0), absl::GetFlag(FLAGS_noise_size), 1, 1}, device);
      torch::Tensor fake_images = generator->forward(noise);
      torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
      torch::Tensor fake_output = discriminator->forward(fake_images.detach());
      torch::Tensor d_loss_fake =
          torch::binary_cross_entropy(fake_output, fake_labels);
      d_loss_fake.backward();

      torch::Tensor d_loss = d_loss_real + d_loss_fake;
      discriminator_optimizer.step();

      // Train generator.
      generator->zero_grad();
      fake_labels.fill_(1);
      fake_output = discriminator->forward(fake_images);
      torch::Tensor g_loss =
          torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();
      generator_optimizer.step();
      batch_index++;
      if (batch_index % absl::GetFlag(FLAGS_log_interval) == 0) {
        std::printf(
            "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f\n",
            epoch,
            absl::GetFlag(FLAGS_number_of_epochs),
            batch_index,
            batches_per_epoch,
            d_loss.item<float>(),
            g_loss.item<float>());
      }

      if (batch_index % absl::GetFlag(FLAGS_checkpoint_every) == 0) {
        // Checkpoint the model and optimizer state.
        torch::save(generator, "generator-checkpoint.pt");
        torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
        torch::save(discriminator, "discriminator-checkpoint.pt");
        torch::save(
            discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
        // Sample the generator and save the images.
        torch::Tensor samples = generator->forward(torch::randn(
            {absl::GetFlag(FLAGS_number_of_samples_per_checkpoint), absl::GetFlag(FLAGS_noise_size), 1, 1}, device));
        torch::save(
            (samples + 1.0) / 2.0,
            torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
        std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
      }
    }
  }

  std::cout << "Training complete!" << std::endl;

}

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  std::cout << "*** Torch ST ***" << std::endl;
//  testModel();
//  testTrainTestMnist();
//  testPoly();
  testDcgan();
  std::cout << "*** Torch EN ***" << std::endl;
  return 0;
}
