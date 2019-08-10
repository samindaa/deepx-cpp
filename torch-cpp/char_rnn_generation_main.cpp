//
// Created by saminda on 8/4/19.
//

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
using ::torch::nn::Dropout;
using ::torch::nn::DropoutOptions;

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

class CharRnnGeneration : public Module {
 public:
  CharRnnGeneration(uint32_t n_categories, uint32_t input_size, uint32_t hidden_size, uint32_t output_size)
      : hidden_size_(hidden_size) {
    i2h_ = register_module("i2h", Linear(n_categories + input_size + hidden_size, hidden_size));
    i2o_ = register_module("i2o", Linear(n_categories + input_size + hidden_size, output_size));
    o2o_ = register_module("o2o", Linear(hidden_size + output_size, output_size));
    dropout_ = register_module("dropout", Dropout(DropoutOptions(0.1)));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor category, torch::Tensor input, torch::Tensor hidden) {
    auto input_combined = torch::cat({category, input, hidden}, 1);
    hidden = i2h_->forward(input_combined);
    auto output = i2o_->forward(input_combined);
    auto output_combined = torch::cat({output, hidden}, 1);
    output = o2o_->forward(output_combined);
    output = dropout_->forward(output);
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
  Linear o2o_{nullptr};
  Dropout dropout_{nullptr};
};

void testCharRnnGeneration() {
  auto filenames = glob(absl::GetFlag(FLAGS_data_pattern));

  std::unordered_map<char, int> all_letters;
  std::unordered_map<int, char> all_letters_inv;
  std::unordered_map<int, std::string> all_categories;
  std::unordered_map<int, std::vector<std::string>> category_lines;
  // Row tensor storage.
  std::unordered_map<std::string, std::vector<int>> category_storage;
  std::unordered_map<char, std::vector<int>> input_storage;
  std::unordered_map<std::string, std::vector<long>> output_storage;

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
  // EOS
  all_letters.emplace(0xFF, all_letters.size());

  for (const auto &e : all_categories) {
    std::vector<int> v(all_categories.size(), 0);
    v[e.first] = 1;
    category_storage.emplace(e.second, v);
  }

  for (const auto &e : all_letters) {
    std::vector<int> v(all_letters.size(), 0);
    v[e.second] = 1;
    input_storage.emplace(e.first, v);
    all_letters_inv.emplace(e.second, e.first);
  }

  // One-hot vector for category.
  auto CategoryTensor = [&category_storage](const std::string &category) {
    return torch::from_blob(category_storage[category].data(),
                            category_storage[category].size(), torch::dtype(torch::kInt).requires_grad(false)).view({1,
                                                                                                                     -1}).to(
        torch::kFloat32);
  };

  auto LetterToTensor = [&input_storage](char c) {
    return torch::from_blob(input_storage[c].data(),
                            input_storage[c].size(), torch::dtype(torch::kInt).requires_grad(false)).view({1,
                                                                                                           -1}).to(torch::kFloat32);
  };

  // One-hot matrix of first to last letters (not including EOS) for input.
  auto InputTensor = [&all_letters, &LetterToTensor](const std::string &line) {
    std::vector<at::Tensor> tensors;
    for (const auto &c : line) {
      tensors.emplace_back(LetterToTensor(c).unsqueeze(0));
    }
    return torch::cat(tensors, 0);
  };

  // LongTensor of second letter to end (EOS) for target.
  auto TargetTensor = [&all_letters, &output_storage, &LetterToTensor](const std::string &line) {
    if (output_storage.find(line) == output_storage.end()) {
      std::vector<long> letter_indexes;
      for (const auto &c : line.substr(1)) {
        letter_indexes.emplace_back(all_letters[c]);
      }
      letter_indexes.emplace_back(all_letters.size() - 1);
      output_storage.emplace(line, letter_indexes);
    }
    auto &letter_indexes = output_storage[line];
    return torch::from_blob(letter_indexes.data(),
                            letter_indexes.size(), torch::dtype(torch::kLong).requires_grad(false));
  };

  int count = 0;
  int test_category = 0;
  std::cout << CategoryTensor(all_categories[test_category]) << std::endl;
  for (const auto &v : category_lines[test_category]) {
    std::cout << v << std::endl;
    std::cout << InputTensor(v).sizes() << std::endl;
    std::cout << TargetTensor(v).sizes() << std::endl;
    if (++count == 5) {
      break;
    }
  }

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()

  // Random item from a list.
  auto RandomChoice = [&gen](size_t max_size) {
    std::uniform_int_distribution<> dis(0, max_size - 1);
    return dis(gen);
  };

  // Get a random category and random line from that category.
  auto RandomTrainingPair = [&all_categories, &category_lines, &RandomChoice]() {
    auto category_index = RandomChoice(all_categories.size());
    auto line_index = RandomChoice(category_lines[category_index].size());
    return std::make_tuple(category_index, line_index);
  };

  // Make category, input, and target tensors from a random category, line pair
  auto RandomTrainingExample =
      [&all_categories, &category_lines, &RandomTrainingPair, &CategoryTensor, &InputTensor, &TargetTensor]() {
        int64_t category_index, line_index;
        std::tie(category_index, line_index) = RandomTrainingPair();
        auto category_tensor = CategoryTensor(all_categories[category_index]);
        auto input_line_tensor = InputTensor(category_lines[category_index][line_index]);
        auto target_line_tensor = TargetTensor(category_lines[category_index][line_index]);
        return std::make_tuple(category_tensor, input_line_tensor, target_line_tensor);
      };

//  for (int i = 0; i < 10; ++i) {
//    size_t category_index, line_index;
//    torch::Tensor category_tensor, line_tensor;
//    std::tie(category_index, line_index, category_tensor, line_tensor) = RandomTrainingExample();
//    std::cout << "category: " << all_categories[category_index] << " line: "
//              << category_lines[category_index][line_index] << " category_tensor: " << category_tensor
//              << " line_tensor: " << line_tensor.sizes() << std::endl;
//  }
//
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

  auto rnn = std::make_shared<CharRnnGeneration>(all_categories.size(), all_letters.size(), 128, all_letters.size());
  rnn->to(device);

  // If you set this too high, it might explode. If too low, it might not learn.
  torch::optim::Adam optimizer(rnn->parameters(), torch::optim::AdamOptions(0.0005));

  auto Train = [&rnn, &device, &optimizer](torch::Tensor category_tensor,
                                           torch::Tensor input_line_tensor,
                                           torch::Tensor target_line_tensor) {
    target_line_tensor.unsqueeze_(-1);
    category_tensor = category_tensor.to(device);
    input_line_tensor = input_line_tensor.to(device);
    target_line_tensor = target_line_tensor.to(device);
    auto hidden = rnn->init_hidden().to(device);
    optimizer.zero_grad();
    torch::Tensor output;
    torch::Tensor loss;
    for (int64_t i = 0; i < input_line_tensor.size(0); ++i) {
      std::tie(output, hidden) = rnn->forward(category_tensor, input_line_tensor[i], hidden);
      auto l = torch::nll_loss(output, target_line_tensor[i]);
      AT_ASSERT(!std::isnan(l.template item<float>()));
      if (i == 0) {
        loss = torch::zeros_like(l);
      }
      loss += l;
    }
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
    torch::Tensor category_tensor, input_line_tensor, target_line_tensor;
    std::tie(category_tensor, input_line_tensor, target_line_tensor) = RandomTrainingExample();
    torch::Tensor output;
    float loss;
    std::tie(output, loss) = Train(category_tensor, input_line_tensor, target_line_tensor);
    current_loss += loss;

    // Print iter number, loss, name, and guess
    if (iter % print_every == 0) {
      std::printf("\n%ld %d%% %.4f",
                  iter,
                  int((float(iter) / n_iters) * 100.0),
                  loss);
    }

    // Add current loss avg to list of losses.
    if (iter % plot_every == 0) {
      all_losses.emplace_back(current_loss / plot_every);
      current_loss = 0;
    }
  }
  std::ofstream archive("char-rnn-generation-checkpoint.pt");
  torch::save(rnn, archive);

  // Sample from a category and starting letter.
  auto Sample =
      [&rnn, &device, &all_letters, &all_letters_inv, &CategoryTensor, &InputTensor](const std::string &category,
                                                                                     const std::string &start_letter = "A",
                                                                                     int64_t max_length = 20) {
        torch::NoGradGuard no_grad;
        rnn->eval();
        auto category_tensor = CategoryTensor(category).to(device);
        auto input = InputTensor(start_letter).to(device);
        auto hidden = rnn->init_hidden().to(device);
        torch::Tensor output;
        std::string output_name = start_letter;
        for (int64_t i = 0; i < max_length; ++i) {
          std::tie(output, hidden) = rnn->forward(category_tensor, input[0], hidden);
          torch::Tensor topv, topi;
          std::tie(topv, topi) = output.topk(1);
          if (topi[0][0].item<int64_t>() == all_letters.size() - 1) {
            break;
          } else {
            auto letter = std::string(1, all_letters_inv[topi[0][0].item<int64_t>()]);
            output_name += letter;
            input = InputTensor(letter).to(device);
          }
        }
        return output_name;
      };

  // Get multiple samples from one category and multiple starting letters.
  auto Samples = [&Sample](const std::string &category, const std::string &start_letters = "ABC") {
    for (const auto &start_letter : start_letters) {
      std::cout << Sample(category, std::string(1, start_letter)) << std::endl;
    }
  };

  Samples("/home/saminda/Data/str_data/names/English.txt", "EML");
  Samples("/home/saminda/Data/str_data/names/Russian.txt", "RUS");
  Samples("/home/saminda/Data/str_data/names/German.txt", "GER");
  Samples("/home/saminda/Data/str_data/names/Spanish.txt", "SPA");
  Samples("/home/saminda/Data/str_data/names/Chinese.txt", "CHI");

};

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  std::cout << "*** Str SRT ***" << std::endl;
  testCharRnnGeneration();
  std::cout << "*** Str END ***" << std::endl;
  return 0;
}
