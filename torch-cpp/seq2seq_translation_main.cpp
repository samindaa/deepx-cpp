//
// Created by saminda on 8/10/19.
//

#include <regex>
#include <vector>
#include <string>
#include <random>
#include <utility>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include <glob.h>
#include <torch/torch.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_split.h"
#include "absl/strings/substitute.h"

using ::torch::nn::Module;
using ::torch::nn::Linear;
using ::torch::nn::Dropout;
using ::torch::nn::DropoutOptions;
using ::torch::nn::Embedding;
using ::torch::nn::EmbeddingOptions;
using ::torch::nn::GRU;
using ::torch::nn::GRUOptions;
using ::torch::nn::RNNOutput;

ABSL_FLAG(std::string, data_root, "/home/saminda/Data/str_data", "Where to find the translation dataset.");

constexpr static size_t kSosToken = 0;
constexpr static size_t kEosToken = 1;

class Lang {

 public:
  explicit Lang(std::string name, size_t max_length)
      : name_(std::move(name)), max_length_(max_length), max_size_(0), num_sentences(0), x_hat_(0), x_hat2_(0) {}

  void AddWord(const std::string &word) {
    if (word2index_.find(word) == word2index_.end()) {
      const auto n_words = word2index_.size();
      word2index_[word] = n_words;
      index2word_[n_words] = word;
    }
    word2count_[word] += 1;
  }

  void AddSentence(const std::string &sentence) {
    std::vector<std::string> words = absl::StrSplit(sentence, absl::ByString(" "));
    for (const auto &word: words) {
      AddWord(word);
    }
    if (words.size() > max_size_) {
      max_size_ = words.size();
    }
    x_hat_ += words.size();
    x_hat2_ += words.size() * words.size();
    ++num_sentences;
  }

  size_t Size() const {
    return word2index_.size();
  }

  size_t MaxLength() const {
    return max_length_;
  }

  std::string Name() const {
    return name_;
  }

  size_t GetWord2Index(const std::string &word) const {
    return word2index_.find(word)->second;
  }

  const std::string &GetIndex2Word(size_t index) const {
    return index2word_.find(index)->second;
  }

  void Stats() const {
    const double mean = x_hat_ / num_sentences;
    const double sigma = (x_hat2_ / num_sentences) - mean * mean;
    std::cout << absl::Substitute("max_size: $0, mean: $1, sigma: $2", max_size_, mean, sigma) << std::endl;
  }

 private:
  std::string name_;
  size_t max_length_;
  size_t max_size_;
  size_t num_sentences;
  double x_hat_;
  double x_hat2_;
  std::unordered_map<std::string, size_t> word2index_;
  std::unordered_map<std::string, size_t> word2count_;
  std::unordered_map<size_t, std::string> index2word_ = {{kSosToken, "SOS"}, {kEosToken, "EOS"}};
};

std::tuple<std::vector<std::string>, std::vector<std::string>> ReadLang(const Lang *input_lang,
    const Lang *output_lang) {
  std::cout << "Reading lines ..." << std::endl;
  std::ifstream infile(absl::Substitute("$0/$1-$2.txt", absl::GetFlag(FLAGS_data_root), input_lang->Name(), output_lang->Name()));
  std::string line;

  std::vector<std::string> input_sentences, output_sentences;
  while (std::getline(infile, line)) {
    if (line.empty()) {
      continue;
    }

    std::vector<std::string> splits = absl::StrSplit(line, absl::MaxSplits('\t', 2));
    assert(splits.size() == 2);

    // TODO(saminda): implement reverse, and normalization
    absl::AsciiStrToLower(&splits[0]);
    absl::AsciiStrToLower(&splits[1]);;

    // Limit lang1 and lang2 lengths.
    if (splits[0].size() < input_lang->MaxLength() && splits[1].size()_< output_lang->MaxLength()) {
      input_sentences.emplace_back(splits[0]);
      output_sentences.emplace_back(splits[1]);
    }
  }

  return {input_sentences, output_sentences};
}

void PrepareData(Lang *input_lang,
                 Lang *output_lang,
                 std::vector<std::string> *lang1_sentences,
                 std::vector<std::string> *lang2_sentences) {
  std::tie(*lang1_sentences, *lang2_sentences) = ReadLang(input_lang, output_lang);
  auto Update = [](Lang *lang, const std::vector<std::string> &sentences) {
    for (const auto &sentence : sentences) {
      lang->AddSentence(sentence);
    }
  };

  Update(input_lang, *lang1_sentences);
  Update(output_lang, *lang2_sentences);

  std::cout << "Counted words: " << std::endl;
  std::cout << input_lang->Name() << " " << input_lang->Size() << std::endl;
  std::cout << output_lang->Name() << " " << output_lang->Size() << std::endl;
}

class EncoderRNN : public Module {
 public:
  EncoderRNN(uint32_t input_size, uint32_t hidden_size)
      : hidden_size_(hidden_size) {
    embedding_ = register_module("embedding", Embedding(EmbeddingOptions(input_size, hidden_size)));
    gru_ = register_module("gru", GRU(GRUOptions(hidden_size, hidden_size)));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input, torch::Tensor hidden) {
    input = embedding_->forward(input).view({1, 1, -1});
    auto output = gru_->forward(input, hidden);
    return {output.output, output.state};
  }

  torch::Tensor InitHidden(torch::Device device) {
    return torch::zeros({1/*num_layers∗num_directions*/, 1/*batch_size*/, hidden_size_}).to(device);
  }

 private:
  uint32_t hidden_size_;
  Embedding embedding_{nullptr};
  GRU gru_{nullptr};
};

class AttnDecoderRNN : public Module {
 public:
  AttnDecoderRNN(uint32_t hidden_size, uint32_t output_size, double dropout_p = 0.1, uint32_t max_length = 10)
      : hidden_size_(hidden_size) {
    embedding_ = register_module("embedding", Embedding(EmbeddingOptions(output_size, hidden_size)));
    attn_ = register_module("attn", Linear(hidden_size * 2, max_length));
    attn_combine_ = register_module("attn_combine", Linear(hidden_size * 2, hidden_size));
    dropout_ = register_module("dropout", Dropout(DropoutOptions(dropout_p)));
    gru_ = register_module("gru", GRU(hidden_size, hidden_size));
    out_ = register_module("out", Linear(hidden_size, hidden_size));
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor input,
                                                                  torch::Tensor hidden,
                                                                  torch::Tensor encoder_outputs) {
    auto embedded = embedding_->forward(input).view({1, 1, -1});
    embedded = dropout_->forward(embedded);

    auto attn_weights = torch::softmax(attn_->forward(torch::cat({embedded[0], hidden[0]}, 1)), 1);
    auto attn_applied = torch::bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0));
    auto output = torch::cat({embedded[0], attn_applied[0]}, 1);
    output = attn_combine_->forward(output).unsqueeze(0);
    output = torch::relu(output);
    auto gru_output = gru_->forward(output, hidden);
    output = torch::log_softmax(out_->forward(output[0]), 1);
    return {output, gru_output.state, attn_weights};
  }

  torch::Tensor InitHidden(torch::Device device) {
    return torch::zeros({1/*num_layers∗num_directions*/, 1/*batch_size*/, hidden_size_}).to(device);
  }

 private:
  uint32_t hidden_size_;
  Embedding embedding_{nullptr};
  Linear attn_{nullptr};
  Linear attn_combine_{nullptr};
  Dropout dropout_{nullptr};
  GRU gru_{nullptr};
  Linear out_{nullptr};
};

class Memory {
 public:

  explicit Memory(Lang *lang) : lang_(lang) {}

  torch::Tensor TensorFromSentence(const std::string &sentence) {
    if (sentence2indices_.find(sentence) == sentence2indices_.end()) {
      std::vector<std::string> words = absl::StrSplit(sentence, absl::ByString(" "));
      std::vector<long> indices;
      for (size_t i = 0; i < words.size(); ++i) {
        if (i < lang_->MaxLength()) {
          indices.emplace_back(lang_->GetWord2Index(words[i]));
        } else {
          break;
        }
      }
      indices.emplace_back(kEosToken);
      sentence2indices_.emplace(sentence, indices);
    }
    return torch::from_blob(sentence2indices_[sentence].data(),
                            sentence2indices_[sentence].size(),
                            torch::dtype(torch::kLong).requires_grad(false)).view({-1, 1});
  }

  Lang *lang_;
  std::unordered_map<std::string, std::vector<long>> sentence2indices_;
};

// Testing
void testSeq2SeqTranslation() {
  std::vector<std::string> lang1_sentences, lang2_sentences;
  uint32_t max_length = 10;
  Lang input_lang("eng", max_length);
  Lang output_lang("fra", max_length);
  PrepareData(&input_lang, &output_lang, &lang1_sentences, &lang2_sentences);
  input_lang.Stats();
  output_lang.Stats();
  std::cout << lang1_sentences.size() << " " << lang2_sentences.size() << std::endl;
  size_t test_idx = 412;
  std::cout << lang1_sentences.at(test_idx) << std::endl;
  std::cout << lang2_sentences.at(test_idx) << std::endl;

  Memory input_memory(&input_lang);
  Memory output_memory(&output_lang);

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> uniform(0.0, 1.0);

  // Random item from a list.
  auto RandomChoice = [&gen](size_t max_size) {
    std::uniform_int_distribution<> dis(0, max_size - 1);
    return dis(gen);
  };

  auto TensorsFromPair = [&input_memory, &output_memory, &lang1_sentences, lang2_sentences](size_t index) {
    auto input_tensor = input_memory.TensorFromSentence(lang1_sentences[index]);
    auto target_tensor = output_memory.TensorFromSentence(lang2_sentences[index]);
    return std::make_tuple(input_tensor, target_tensor);
  };

  auto test_tup = TensorsFromPair(test_idx);
  std::cout << std::get<0>(test_tup) << std::endl;
  std::cout << std::get<1>(test_tup) << std::endl;

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  const uint32_t hidden_size = 256;
  auto encoder = std::make_shared<EncoderRNN>(input_lang.Size(), hidden_size);
  encoder->to(device);
  auto decoder = std::make_shared<AttnDecoderRNN>(hidden_size, output_lang.Size());
  decoder->to(device);

  size_t training_size = lang1_sentences.size();
  uint32_t n_iters = 75000;
  uint32_t print_every = 5000;
  uint32_t plot_every = 100;
  double learning_rate = 0.01;
  double teacher_forcing_ratio = 0.5;

  auto Train =
      [&encoder, &decoder, &uniform, &gen, &max_length, &hidden_size, &teacher_forcing_ratio, &device](torch::Tensor input_tensor,
                                                                                                       torch::Tensor target_tensor,
                                                                                                       torch::optim::Adam &encoder_optimizer,
                                                                                                       torch::optim::Adam &decoder_optimizer) {
        auto encoder_hidden = encoder->InitHidden(device);
        encoder_optimizer.zero_grad();
        decoder_optimizer.zero_grad();

        auto input_length = input_tensor.size(0);
        auto target_length = target_tensor.size(0);

        auto encoder_outputs = torch::zeros({max_length, hidden_size}).to(device);
        torch::Tensor encoder_output;
        for (int64_t ei = 0; ei < input_length; ++ei) {
          std::tie(encoder_output, encoder_hidden) = encoder->forward(input_tensor[ei], encoder_hidden);
          encoder_outputs[ei] = encoder_output[0][0];
        }

        auto decoder_input = torch::tensor({(int64_t) kSosToken}).to(device);
        auto decoder_hidden = encoder_hidden;

        auto use_teacher_forcing = uniform(gen) < teacher_forcing_ratio;
        torch::Tensor decoder_output, decoder_attention;
        torch::Tensor loss;
        if (use_teacher_forcing) {
          // Teacher forcing: Feed the target as the next input.
          for (int64_t di = 0; di < target_length; ++di) {
            std::tie(decoder_output, decoder_hidden, decoder_attention) =
                decoder->forward(decoder_input, decoder_hidden, encoder_outputs);
            auto l = torch::nll_loss(decoder_output, target_tensor[di]);
            AT_ASSERT(!std::isnan(l.template item<float>()));
            if (di == 0) {
              loss = torch::zeros_like(l);
            }
            loss += l;
            decoder_input = target_tensor[di];  // Teacher forcing
          }
        } else {
          // Without teacher forcing: use its own predictions as the next input.
          for (int64_t di = 0; di < target_length; ++di) {
            std::tie(decoder_output, decoder_hidden, decoder_attention) =
                decoder->forward(decoder_input, decoder_hidden, encoder_outputs);

            torch::Tensor topv, topi;
            std::tie(topv, topi) = decoder_output.topk(1);
            decoder_input = topi.squeeze().detach(); // detach from history as input

            auto l = torch::nll_loss(decoder_output, target_tensor[di]);
            AT_ASSERT(!std::isnan(l.template item<float>()));
            if (di == 0) {
              loss = torch::zeros_like(l);
            }
            loss += l;

            if (decoder_input.item<int64_t>() == kEosToken) {
              break;
            }
          }
        }

        loss.backward();
        encoder_optimizer.step();
        decoder_optimizer.step();

        return loss.template item<float>() / target_length;
      };

  auto TrainIters =
      [&encoder, &decoder, &RandomChoice, &TensorsFromPair, &Train, &training_size](uint32_t n_iters,
                                                                                    uint32_t print_every,
                                                                                    uint32_t plot_every,
                                                                                    double learning_rate) {
        torch::optim::Adam encoder_optimizer(encoder->parameters(), torch::optim::AdamOptions(learning_rate));
        torch::optim::Adam decoder_optimizer(decoder->parameters(), torch::optim::AdamOptions(learning_rate));

        std::vector<float> plot_losses;
        float print_loss_total = 0;  // Reset every print_every
        float plot_loss_total = 0;  // Reset every plot_every
        for (int64_t iter = 1; iter <= n_iters; ++iter) {
          torch::Tensor input_tensor, target_tensor;
          std::tie(input_tensor, target_tensor) = TensorsFromPair(RandomChoice(training_size));
          auto loss = Train(input_tensor, target_tensor, encoder_optimizer, decoder_optimizer);
          print_loss_total += loss;
          plot_loss_total += loss;

          // Print iter number, loss, name, and guess
          if (iter % print_every == 0) {
            print_loss_total = 0
            std::printf("\n%ld %d%% %.4f",
                        iter,
                        int((float(iter) / n_iters) * 100.0),
                        print_loss_total);
          }

          // Add current loss avg to list of losses.
          if (iter % plot_every == 0) {
            plot_losses.emplace_back(plot_loss_total / plot_every);
            plot_loss_total = 0;
          }
        }
      };

  TrainIters(n_iters, print_every, plot_every, learning_rate);

}

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  std::cout << "*** Str SRT ***" << std::endl;
  testSeq2SeqTranslation();
  std::cout << "*** Str END ***" << std::endl;
  return 0;
}
