//
// Created by saminda on 8/3/19.
//
// ./deepx_cpp --data_pattern="./data/names/*.txt"

#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include <glob.h>
#include <torch/torch.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/substitute.h"

using ::torch::nn::Module;
using ::torch::nn::Linear;

ABSL_FLAG(std::string, data_pattern, "/home/saminda/Data/str_data/names/*.txt", "Where to find the names dataset");

std::vector<std::string> glob(const std::string &pattern) {
  // Glob struct resides on the stack.
  glob_t glob_result;
  memset(&glob_result, 0, sizeof(glob_result));

  // Do the glob operation.
  auto return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  if (return_value != 0) {
    globfree(&glob_result);
    throw std::runtime_error(absl::Substitute("glob() failed with return value: $0", return_value));
  }

  // Collect all the filenames into a list.
  std::vector<std::string> filenames;
  for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
    filenames.emplace_back(std::string(glob_result.gl_pathv[i]));
  }
  return filenames;
}

class Rnn : public Module {
 public:
  Rnn(uint32_t input_size, uint32_t hidden_size, uint32_t output_size) : hidden_size_(hidden_size) {
    i2h_ = register_module("i2h", Linear(input_size + hidden_size, hidden_size));
    i2o_ = register_module("i2o", Linear(input_size + hidden_size, output_size));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input, torch::Tensor hidden) {
    auto combined = torch::cat({input, hidden}, 1);
    hidden = i2h_->forward(combined);
    auto output = i2o_->forward(combined);
    output = torch::log_softmax(output, 1);
    return {output, hidden};
  }

  torch::Tensor init_hidden() {
    return torch::zeros({1, hidden_size_});
  }

 private:
  uint32_t hidden_size_;
  Linear i2h_{nullptr};
  Linear i2o_{nullptr};
};

void testRnn() {
  auto filenames = glob(absl::GetFlag(FLAGS_data_pattern));

  std::unordered_map<char, int> all_letters;
  std::unordered_map<int, std::string> all_categories;
  std::unordered_map<int, std::vector<std::string>> category_lines;
  std::unordered_map<char, std::vector<int>> storage;

  std::string line;
  for (const auto &filename : filenames) {
    std::ifstream infile(filename);
    const auto category_idx = all_categories.size();
    all_categories.emplace(category_idx, filename);
    while (std::getline(infile, line)) {
      if (line.empty()) {
        continue;
      }
      category_lines[category_idx].emplace_back(line);
      for (const auto &c : line) {
        if (all_letters.find(c) == all_letters.end()) {
          all_letters.emplace(c, all_letters.size());
        }
      }
    }
  }

  for (const auto &e : all_letters) {
    std::vector<int> v(all_letters.size(), 0);
    v[e.second] = 1;
    storage.emplace(e.first, v);
  }

  auto LetterToTensor = [&all_letters, &storage](char c) {
    return torch::from_blob(storage[c].data(),
                            storage[c].size(), torch::dtype(torch::kInt).requires_grad(false)).view({1,
                                                                                                     -1}).to(torch::kFloat32);
  };

  auto LineToTensor = [&all_letters, &LetterToTensor](const std::string &line) {
    std::vector<at::Tensor> tensors;
    for (const auto &c : line) {
      tensors.emplace_back(LetterToTensor(c).unsqueeze(0));
    }
    return torch::cat(tensors, 0);
  };

  int count = 0;
  for (const auto &v : category_lines[0]) {
    std::cout << v << std::endl;
    std::cout << LineToTensor(v).sizes() << std::endl;
    if (++count == 5) {
      break;
    }
  }

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()

  auto RandomChoice = [&gen](size_t max_size) {
    std::uniform_int_distribution<> dis(0, max_size - 1);
    return dis(gen);
  };

  auto RandomTrainingExample = [&all_categories, &category_lines, &LineToTensor, &RandomChoice]() {
    auto category_index = RandomChoice(all_categories.size());
    auto line_index = RandomChoice(category_lines[category_index].size());
    auto category_tensor = torch::tensor(category_index, torch::kLong);
    auto line_tensor = LineToTensor(category_lines[category_index][line_index]);
    return std::make_tuple(category_index, line_index, category_tensor, line_tensor);
  };

  for (int i = 0; i < 10; ++i) {
    size_t category_index, line_index;
    torch::Tensor category_tensor, line_tensor;
    std::tie(category_index, line_index, category_tensor, line_tensor) = RandomTrainingExample();
    std::cout << "category: " << all_categories[category_index] << " line: "
              << category_lines[category_index][line_index] << " category_tensor: " << category_tensor
              << " line_tensor: " << line_tensor.sizes() << std::endl;
  }

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
//  torch::Device device(device_type);

  // Simple enough to train on CPUs.
  torch::Device device(torch::kCPU);

  auto rnn = std::make_shared<Rnn>(all_letters.size(), 128, all_categories.size());
  rnn->to(device);

  // If you set this too high, it might explode. If too low, it might not learn.
  torch::optim::SGD optimizer(rnn->parameters(), torch::optim::SGDOptions(0.005));

  auto Train = [&rnn, &device, &optimizer](torch::Tensor category_tensor, torch::Tensor line_tensor) {
    category_tensor = category_tensor.to(device);
    line_tensor = line_tensor.to(device);
    auto hidden = rnn->init_hidden().to(device);
    optimizer.zero_grad();
    torch::Tensor output;
    for (int64_t i = 0; i < line_tensor.size(0); ++i) {
      std::tie(output, hidden) = rnn->forward(line_tensor[i], hidden);
    }
    auto loss = torch::nll_loss(output, category_tensor);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    return std::make_tuple(output, loss.template item<float>());
  };

  int64_t n_iters = 100000, print_every = 5000, plot_every = 1000;
  // Keep track of losses for plotting.
  float current_loss = 0;
  std::vector<float> all_losses;
  rnn->train();
  for (int64_t iter = 1; iter <= n_iters; ++iter) {
    size_t category_index, line_index;
    torch::Tensor category_tensor, line_tensor;
    std::tie(category_index, line_index, category_tensor, line_tensor) = RandomTrainingExample();
    torch::Tensor output;
    float loss;
    std::tie(output, loss) = Train(category_tensor, line_tensor);
    current_loss += loss;

    // Print iter number, loss, name, and guess
    if (iter % print_every == 0) {
      auto guess = output.argmax(1).item<int64_t>();
      std::string pred = "✓";
      if (guess != category_index) {
        pred = absl::Substitute("✗ $0", all_categories[guess]);
      }
      std::printf("\n%ld %d%% %.4f %s / %s %s",
                  iter,
                  int((float(iter) / n_iters) * 100.0),
                  loss,
                  category_lines[category_index][line_index].c_str(),
                  all_categories[guess].c_str(),
                  pred.c_str());
    }

    // Add current loss avg to list of losses.
    if (iter % plot_every == 0) {
      all_losses.emplace_back(current_loss / plot_every);
      current_loss = 0;
    }
  }
  std::ofstream archive("char-rnn-classification-checkpoint.pt");
  torch::save(rnn, archive);

  // Just return an output given a line.
  auto Evaluate = [&rnn, &device](torch::Tensor line_tensor) {
    line_tensor = line_tensor.to(device);
    auto hidden = rnn->init_hidden().to(device);
    torch::Tensor output;
    for (int64_t i = 0; i < line_tensor.size(0); ++i) {
      std::tie(output, hidden) = rnn->forward(line_tensor[i], hidden);
    }
    return output;
  };

  std::cout << std::endl;
  int64_t n_predictions = 3;
  n_iters = 20;
  torch::NoGradGuard no_grad;
  rnn->eval();
  for (int iter = 1; iter <= n_iters; ++iter) {
    size_t category_index, line_index;
    torch::Tensor category_tensor, line_tensor;
    std::tie(category_index, line_index, category_tensor, line_tensor) = RandomTrainingExample();
    auto output = Evaluate(line_tensor);

    torch::Tensor topv, topi;
    std::tie(topv, topi) = output.topk(n_predictions, 1);

    std::cout << absl::Substitute("$0 > $1", iter, category_lines[category_index][line_index]) << std::endl;
    for (int64_t j = 0; j < n_predictions; ++j) {
      auto predicted_index = topi[0][j].item<int64_t>();
      std::cout << absl::Substitute("$0", all_categories[predicted_index]) << std::endl;
    }
  }
}

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  std::cout << "*** Str SRT ***" << std::endl;
  testRnn();
  std::cout << "*** Str END ***" << std::endl;
  return 0;
}
