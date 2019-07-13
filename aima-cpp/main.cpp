#include <iostream>
#include <fstream>
#include <random>
#include <unordered_map>

#include "../abseil-cpp/absl/flags/flag.h"
#include "../abseil-cpp/absl/strings/substitute.h"
//#include "abseil-cpp/absl/container/flat_hash_map.h"

#include "search4e.h"
#include "tiles3.h"

ABSL_FLAG(int, count, 0, "Loop count");

void testNode() {
//  aima_cpp::Node failure("failure");
//  std::cout << "state:" << failure.state << std::endl;
}

void testRouteProblem() {
  aima_cpp::Map romania(
      {{{"O", "Z"}, 71}, {{"O", "S"}, 151}, {{"A", "Z"}, 75}, {{"A", "S"}, 140},
       {{"A", "T"}, 118},
       {{"L", "T"}, 111}, {{"L", "M"}, 70}, {{"D", "M"}, 75}, {{"C", "D"}, 120},
       {{"C", "R"}, 146},
       {{"C", "P"}, 138}, {{"R", "S"}, 80}, {{"F", "S"}, 99}, {{"B", "F"}, 211},
       {{"B", "P"}, 101},
       {{"B", "G"}, 90}, {{"B", "U"}, 85}, {{"H", "U"}, 98}, {{"E", "H"}, 86},
       {{"U", "V"}, 142},
       {{"I", "V"}, 92}, {{"I", "N"}, 87}, {{"P", "R"}, 97}},
      {{"A", {76, 497}}, {"B", {400, 327}}, {"C", {246, 285}},
       {"D", {160, 296}}, {"E", {558, 294}},
       {"F", {285, 460}}, {"G", {368, 257}}, {"H", {548, 355}},
       {"I", {488, 535}}, {"L", {162, 379}},
       {"M", {160, 343}}, {"N", {407, 561}}, {"O", {117, 580}},
       {"P", {311, 372}}, {"R", {227, 412}},
       {"S", {187, 463}}, {"T", {83, 414}}, {"U", {471, 363}},
       {"V", {535, 473}}, {"Z", {92, 539}}});

  aima_cpp::PriorityQueue pq(aima_cpp::g);
//  aima_cpp::RouteProblem r0("A", "A", romania);
//  aima_cpp::RouteProblem r1("A", "B", romania);
  aima_cpp::RouteProblem r2("N", "L", romania);
//  aima_cpp::RouteProblem r3("E", "T", romania);
//  aima_cpp::RouteProblem r4("O", "M", romania);

//  for (const auto &kv: romania.neighbors) {
//    std::cout << absl::Substitute("state: $0 ", kv.first) << absl::Substitute(
//        "neighbors: $0",
//        absl::StrJoin(kv.second, " ")) << std::endl;
//  }
//
//  for (const auto &kv: romania.locations) {
//    pq.Add(std::make_shared<aima_cpp::Node>(kv.first,
//                                            nullptr,
//                                            "",
//                                            kv.second.second));
//  }
//
//  while (!pq.empty()) {
//    auto node = pq.Pop();
//    std::cout << absl::Substitute("state: $0, priority: $1",
//                                  node->state,
//                                  aima_cpp::g(node)) << std::endl;
//  }

  auto result = aima_cpp::UniformCostSearch(r2);
  std::vector<std::string> path_states;
  aima_cpp::PathStates(result, path_states);
  std::cout << absl::Substitute("$0", absl::StrJoin(path_states, " "))
            << std::endl;
}

void testIHT() {
  IHT iht(1024);
  TileCoder tc(iht);
  auto test_f = [&tc](const std::vector<float> &floats) {
    auto indices = tc.tiles(8, floats);
    std::cout << absl::Substitute("indices: $0", absl::StrJoin(indices, ","))
              << std::endl;
  };
  test_f({3.6, 7.21});
  test_f({3.7, 7.21}); // a nearby point
  test_f({4, 7}); // while a farther away point
  test_f({-37.2, 7}); // and a point more than one away in any dim
}

void testTileCoder() {
  IHT iht(2048);
  TileCoder tc(iht);
  std::vector<float> weights(2048, 0);
  const float scale_factor = 4. / (3 - 1);
  const float step_size = 0.1 / 8;

  auto mytiles = [&tc, &scale_factor](const float &x) {
    return tc.tiles(8, {x * scale_factor});
  };

  auto learn =
      [&weights, &tc, &mytiles, &step_size](const float &x, const float &z) {
        const auto tiles = mytiles(x);
        float estimate = 0.;
        for (const auto &tile : tiles) {
          estimate += weights[tile];
        }
        const float error = z - estimate;
        for (const auto &tile : tiles) {
          weights[tile] += step_size * error;
        }
      };

  std::unordered_map<float, float> training_data;
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 eng(rd()); // seed the generator
  std::uniform_real_distribution<> dist(1, 3); // define the range
  for (int n = 0; n < 100; ++n) {
    const float x = dist(eng);
    training_data.emplace(x, x * x);
  }

  auto test = [&weights, &tc, &mytiles](const float &x) {
    float estimate = 0.;
    const auto tiles = mytiles(x);
    for (const auto &tile : tiles) {
      estimate += weights[tile];
    }
    return estimate;
  };

  for (int epoch = 0; epoch < 50; ++epoch) {
    for (const auto &e: training_data) {
      learn(e.first, e.second);
    }
    if (epoch % 10 == 0) {
      std::cout << absl::Substitute("epoch: $0 done", epoch) << std::endl;
    }
  }

  // TEST
  std::ofstream ofs("test.csv");
  ofs << "x,z,pred\n";
  for (float x = 1.; x < 3; x += 0.05) {
    const auto estimate = test(x);
    ofs << absl::Substitute("$0,$1,$2\n",
                            x,
                            x * x,
                            test(x));
  }
}

class FunctionLearner {
 public:
  IHT iht_;
  TileCoder tc_;
  std::vector<float> w_;
  int num_tilings_;
  float scale_factor_;
  float step_size_;

  FunctionLearner(const int &size,
                  const int &num_tilings,
                  const std::pair<float, float> &range)
      : iht_(size),
        tc_(iht_),
        num_tilings_(num_tilings),
        scale_factor_(4.0 / (range.second - range.first)),
        step_size_(0.1 / num_tilings) {
    w_.resize(size, 0.);
  }

  std::vector<int> mytiles(const float &x) {
    return tc_.tiles(num_tilings_, {x * scale_factor_});
  }

  void learn(const float &x, const float &z) {
    const auto tiles = mytiles(x);
    float estimate = 0.;
    for (const auto &tile : tiles) {
      estimate += w_[tile];
    }
    const float error = z - estimate;
    for (const auto &tile : tiles) {
      w_[tile] += step_size_ * error;
    }
  }

  float test(const float &x) {
    float estimate = 0.;
    const auto tiles = mytiles(x);
    for (const auto &tile : tiles) {
      estimate += w_[tile];
    }
    return estimate;
  }

  void train_test(const int &idx,
                  const std::function<float(float)> &function,
                  const std::pair<float, float> &range,
                  const int &num_samples = 200,
                  const int &num_epoch = 30) {

    std::unordered_map<float, float> training_data;
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_real_distribution<>
        dist(range.first, range.second); // define the range
    for (int n = 0; n < num_samples; ++n) {
      const float x = dist(eng);
      training_data.emplace(x, function(x));
    }

    for (int epoch = 0; epoch < num_epoch; ++epoch) {
      for (const auto &e: training_data) {
        learn(e.first, e.second);
      }
      if (epoch % 10 == 0) {
        std::cout << absl::Substitute("epoch: $0 done", epoch) << std::endl;
      }
    }

    // TEST
    const float step_size = (range.second - range.first) / 50.;
    std::cout << absl::Substitute("Testing: $0 -> [$1, $2] step_size: $3",
                                  idx,
                                  range.first,
                                  range.second,
                                  step_size) << std::endl;
    std::ofstream ofs(absl::Substitute("function_test_$0.csv", idx));
    ofs << "x,z,pred\n";
    for (float x = range.first; x < range.second; x += step_size) {
      const auto estimate = test(x);
      ofs << absl::Substitute("$0,$1,$2\n",
                              x,
                              function(x),
                              test(x));
    }
  }
};

void testFunctions() {
  std::unordered_map<int, std::function<float(float)>> test_functions;
  std::unordered_map<int, std::pair<float, float>> test_ranges;

  auto function_and_range =
      [&test_functions, &test_ranges](std::function<float(float)> function,
                                      std::pair<float, float> range) {
        test_functions.emplace(test_functions.size(), function);
        test_ranges.emplace(test_ranges.size(), range);
      };

  function_and_range([](float x) {
    return std::sin(x) + sin(x * 10 / 3);
  }, {2.7, 7.5});

  function_and_range([](float x) {
    float sum = 0.;
    for (int k = 1; k <= 6; ++k) {
      sum += k * std::sin((k + 1) * x + k);
    }
    return -sum;
  }, {-10, 10});

  function_and_range([](float x) {
    return -(16 * std::pow(x, 2) - 24 * x + 5) * std::exp(-x);
  }, {1.9, 3.9});

  function_and_range([](float x) { return -(1.4 - 3 * x) * std::sin(18 * x); },
                     {0, 1.2});

  function_and_range([](float x) {
    return -(x + std::sin(x)) * std::exp(-std::pow(x, 2));
  }, {-10, 10});

  function_and_range([](float x) {
    return std::sin(x) + sin(x * 10 / 3) + std::log(x) - 0.84 * x + 3;
  }, {2.7, 7.5});

  function_and_range([](float x) {
    float sum = 0.;
    for (int k = 1; k <= 6; ++k) {
      sum = k * std::cos((k + 1) * x + k);
    }
    return -sum;
  }, {-10, 10});

  function_and_range([](float x) { return std::sin(x) + std::sin(x * 2 / 3); },
                     {3.1, 20.4});

  function_and_range([](float x) { return -x * std::sin(x); }, {0, 10});

  function_and_range([](float x) { return 2 * std::cos(x) + std::cos(2 * x); },
                     {-M_PI_2, M_PI_2});

  function_and_range([](float x) {
    return std::pow(std::sin(x), 3) + std::pow(std::cos(x), 3);
  }, {0, M_PI * 2});

  function_and_range([](float x) {
    return -std::pow(x, 2. / 3) - std::pow((1 - std::pow(x, 2)), 1. / 3);
  }, {0.001, 0.99});

  function_and_range([](float x) {
    return -std::exp(-x) * std::sin(2 * M_PI * x);
  }, {0, 4});

  function_and_range([](float x) { return (x * x - 5 * x + 6) / (x * x + 1); },
                     {-5, 5});

  function_and_range([](float x) {
    return x <= 3 ? std::pow((x - 2), 2) : 2 * std::log(x - 2) + 1;
  }, {0, 6});

  function_and_range([](float x) {
    return -(x - std::sin(x)) * std::exp(-x * x);
  }, {-10, 10});

  function_and_range([](float x) {
    return x * std::sin(x) + x * std::cos(2 * x);
  }, {0, 10});

  function_and_range([](float x) {
    return std::exp(-3 * x) - std::pow(std::sin(x), 3);
  }, {0, 20});

  for (size_t i = 0; i < test_functions.size(); ++i) {
    std::cout << absl::Substitute("Learning: $0 -> [$1, $2]",
                                  i,
                                  test_ranges[i].first,
                                  test_ranges[i].second) << std::endl;
    FunctionLearner fl(2048, 8, test_ranges[i]);
    fl.train_test(i, test_functions[i], test_ranges[i]);
  }
}

int main() {
  std::cout << absl::Substitute("Abseil: $0", absl::GetFlag(FLAGS_count))
            << std::endl;
//  testNode();
//  testRouteProblem();
//  testIHT();
//  testTileCoder();
  testFunctions();
  return 0;
}