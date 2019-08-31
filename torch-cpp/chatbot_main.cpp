#include <utility>

//
// Created by saminda on 8/17/19.
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

ABSL_FLAG(std::string,
          data_root,
          "/home/saminda/Data/cornell_movie_dialogs_corpus",
          "Where to find the Cornell Movie-Dialogs Corpus.");

// Default word tokens.
constexpr static size_t kPadToken = 0; // Used for padding short sentences.
constexpr static size_t kSosToken = 1; // Start-of-sentence token.
constexpr static size_t kEosToken = 2; // End-of-sentence token.
constexpr static size_t kMaxLength = 10; // Maximum sentence length to consider.
constexpr static size_t kMinCount = 3; // Minimum word count threshold for trimming.
const static std::unordered_map<size_t, std::string> *kIndex2word =
    new std::unordered_map<size_t, std::string>{{kPadToken, "PAD"}, {kSosToken, "SOS"}, {kEosToken, "EOS"}};

struct Line {
  std::string line_id;
  std::string character_id;
  std::string movie_id;
  std::string character;
  std::string text;

  Line(std::string line_id,
       std::string character_id,
       std::string movie_id,
       std::string character,
       std::string text)
      : line_id(std::move(line_id)),
        character_id(std::move(character_id)),
        movie_id(std::move(movie_id)),
        character(std::move(character)),
        text(std::move(text)) {}
};

struct Conversation {
  std::string character1_id;
  std::string character2_id;
  std::string movie_id;
  std::vector<std::string> utterance_ids;

  Conversation(std::string character1_id,
               std::string character2_id,
               std::string movie_id)
      : character1_id(std::move(character1_id)),
        character2_id(std::move(character2_id)),
        movie_id(std::move(movie_id)) {}
};

// Splits each line of the file into a dictionary of fields.
std::unordered_map<std::string, std::shared_ptr<Line>> LoadLines(const std::string &filename) {
  std::ifstream infile(absl::Substitute("$0/$1", absl::GetFlag(FLAGS_data_root), filename));
  std::string line;
  std::unordered_map<std::string, std::shared_ptr<Line>> ret;
  while (std::getline(infile, line)) {
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> splits = absl::StrSplit(line, absl::MaxSplits(absl::ByString(" +++$+++ "), 5));
    assert(splits.size() == 5);
    ret.emplace(std::make_pair(splits[0],
                               std::make_shared<Line>(splits[0], splits[1], splits[2], splits[3], splits[4])));
  }
  return ret;
}

// Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
std::vector<std::shared_ptr<Conversation>> LoadConversations(const std::string &filename) {
  std::ifstream infile(absl::Substitute("$0/$1", absl::GetFlag(FLAGS_data_root), filename));
  std::smatch match;
  std::regex r("L[0-9]+");
  std::string line;
  std::vector<std::shared_ptr<Conversation>> ret;
  while (std::getline(infile, line)) {
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> splits = absl::StrSplit(line, absl::MaxSplits(absl::ByString(" +++$+++ "), 4));
    assert(splits.size() == 4);
    auto conversation = std::make_shared<Conversation>(splits[0], splits[1], splits[2]);
    std::string s(splits[3]);
    while (std::regex_search(s, match, r)) {
      conversation->utterance_ids.emplace_back(match.str(0));
      s = match.suffix().str();
    }
    ret.emplace_back(conversation);
  }
  return ret;
}

// Lowercase, trim, and remove non-letter characters.
std::string NormalizeString(const std::string &s) {
  std::string tmp(s);
  absl::AsciiStrToLower(&tmp);
  std::regex words_regex("[a-zA-Z]+");
  auto words_begin = std::sregex_iterator(tmp.begin(), tmp.end(), words_regex);
  auto words_end = std::sregex_iterator();
  std::vector<std::string> matches;
  for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
    std::smatch match = *i;
    std::string match_str = match.str();
    matches.emplace_back(match_str);
  }
  return absl::StrJoin(matches, " ");
}

// Extracts pairs of sentences from conversations.
std::vector<std::pair<std::string, std::string>> ExtractSentencePairs(
    const std::unordered_map<std::string, std::shared_ptr<Line>> &lines,
    const std::vector<std::shared_ptr<Conversation>> &conversations) {
  std::vector<std::pair<std::string, std::string>> ret;
  for (const auto &conversation : conversations) {
    // Iterate over all the lines of the conversation
    for (size_t i = 0; i < conversation->utterance_ids.size() - 1; ++i) {
      ret.emplace_back(std::make_pair(lines.find(conversation->utterance_ids[i])->second->text,
                                      lines.find(conversation->utterance_ids[i + 1])->second->text));
    }
  }
  return ret;
}

class Voc {

 public:
  explicit Voc(std::string name)
      : name_(std::move(name)), trimmed_(false), max_size_(0), num_sentences(0), x_hat_(0), x_hat2_(0) {
    Initialize();
  }

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

  // Remove words below a certain count threshold.
  void Trim(size_t min_count) {
    if (trimmed_) {
      return;
    }
    trimmed_ = true;
    std::vector<std::string> keep_words;
    for (const auto &kv : word2count_) {
      if (kv.second >= min_count) {
        keep_words.emplace_back(kv.first);
      }
    }
    Initialize();
    for (const auto &word : keep_words) {
      AddWord(word);
    }
  }

  size_t Size() const {
    return word2index_.size();
  }

  std::string Name() const {
    return name_;
  }

  size_t GetWord2Index(const std::string &word) const {
    return word2index_.find(word)->second;
  }

  bool ContainsWord(const std::string &word) const {
    return word2index_.find(word) != word2count_.end();
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
  void Initialize() {
    word2index_.clear();
    word2count_.clear();
    index2word_.clear();
    for (const auto &kv : *kIndex2word) {
      index2word_.emplace(kv.first, kv.second);
    }
    max_size_ = 0;
    num_sentences = 0;
    x_hat_ = 0;
    x_hat2_ = 0;
  }

 private:
  std::string name_;
  bool trimmed_;
  size_t max_size_;
  size_t num_sentences;
  double x_hat_;
  double x_hat2_;
  std::unordered_map<std::string, size_t> word2index_;
  std::unordered_map<std::string, size_t> word2count_;
  std::unordered_map<size_t, std::string> index2word_;
};

void PrepareData(Voc *voc,
                 std::vector<std::pair<std::string, std::string>> *dst,
                 const std::vector<std::pair<std::string, std::string>> &src) {
  std::vector<std::pair<std::string, std::string>> tmp_dest;
  for (const auto &pair : src) {
    std::string first_str = NormalizeString(pair.first);
    std::string second_str = NormalizeString(pair.second);
    if (first_str.size() < kMaxLength && second_str.size() < kMaxLength) {
      tmp_dest.emplace_back(std::make_pair(first_str, second_str));
      voc->AddSentence(first_str);
      voc->AddSentence(second_str);
    }
  }

  std::cout << "Src: " << src.size() << std::endl;
  std::cout << "Voc: " << voc->Size() << std::endl;
  voc->Stats();
  voc->Trim(kMinCount);
  std::cout << "Voc trim: " << voc->Size() << std::endl;

  auto Keep = [](Voc *voc, const std::string &s) {
    bool keep = true;
    std::vector<std::string> tmp = absl::StrSplit(s, absl::ByString(" "));
    for (const auto &t : tmp) {
      if (!voc->ContainsWord(t)) {
        keep = false;
        break;
      }
    }
    return keep;
  };

  for (const auto &pair : tmp_dest) {
    if (Keep(voc, pair.first) && Keep(voc, pair.second)) {
      if (!pair.first.empty() && !pair.second.empty()) {
        dst->emplace_back(std::make_pair(pair.first, pair.second));
        voc->AddSentence(pair.first);
        voc->AddSentence(pair.second);
      }
    }
  }
  std::cout << "Dst: " << dst->size() << std::endl;
  std::cout << "Voc: " << voc->Size() << std::endl;
  voc->Stats();
}

void LoadFormattedLines() {

}

// Testing
void TestLoadLinesLoadConversationsExtractSentencePairs() {
  auto lines = LoadLines("movie_lines.txt");
  std::cout << lines.size() << std::endl;

  auto PrintLine = [](const Line &line) {
    std::cout << line.line_id << " " << line.character_id << " " << line.movie_id << " " << line.character << " "
              << line.text << std::endl;
  };
  PrintLine(*lines["L194"]);
  PrintLine(*lines["L195"]);
  PrintLine(*lines["L196"]);
  PrintLine(*lines["L197"]);

  auto conversations = LoadConversations("movie_conversations.txt");
  std::cout << conversations.size() << std::endl;
  auto PrintConversation = [](const Conversation &conversation) {
    std::cout << conversation.character1_id << " " << conversation.character2_id << " " << conversation.movie_id
              << std::endl;
    for (const auto &utt : conversation.utterance_ids) {
      std::cout << utt << " ";
    }
    std::cout << std::endl;
  };
  PrintConversation(*conversations[0]);
  PrintConversation(*conversations[1]);

  const auto pairs = ExtractSentencePairs(lines, conversations);
  std::cout << pairs.size() << std::endl;
  auto PrintPair = [](const std::pair<std::string, std::string> &pair) {
    std::cout << pair.first << " => " << pair.second << std::endl;
  };
  PrintPair(pairs[0]);
  std::cout << NormalizeString(pairs[0].first) << std::endl;
  std::cout << NormalizeString(pairs[0].second) << std::endl;

  Voc voc("cornell_movie_dialogs_corpus");
  std::vector<std::pair<std::string, std::string>> dst;
  PrepareData(&voc, &dst, pairs);
  PrintPair(dst[0]);
  PrintPair(dst[100]);
  PrintPair(dst[200]);

  std::ofstream out(absl::Substitute("$0/$1",  absl::GetFlag(FLAGS_data_root), "formatted_movie_lines.txt"));
  for (const auto& pair : dst) {
    out << pair.first << "\t" << pair.second << "\n";
  }
  out.close();
}

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  std::cout << "*** Str SRT ***" << std::endl;
  TestLoadLinesLoadConversationsExtractSentencePairs();
  std::cout << "*** Str END ***" << std::endl;
  return 0;
}
