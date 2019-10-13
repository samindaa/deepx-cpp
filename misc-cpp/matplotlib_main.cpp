//
// Created by saminda on 8/31/19.
//
#include <cmath>
#include <vector>

#include <torch/torch.h>

#include "matplotlibcpp.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/substitute.h"

namespace plt = matplotlibcpp;

void TestMinimalExample() {
  plt::plot({1, 3, 2, 4});
  plt::show();
}

void TestComprehensiveExample() {
  const int n = 5000;
  std::vector<double> x(n), y(n), z(n), w(n, 2);
  for (int i = 0; i < n; ++i) {
    x[i] = i * i;
    y[i] = std::sin(2. * M_PI * i/ 360.);
    z[i] = std::log(i);
  }

  plt::figure_size(1200, 780);
  plt::plot(x, y);
  plt::plot(x, w, "r--");
  plt::named_plot("log(x)", x, z);
  plt::xlim(0, 1000 * 1000);
  plt::title("Sample figgure");
  plt::legend();
  plt::show();
  //plt::save("/Tmp/basic.png");
}

void Test3DFunction() {
  std::vector<std::vector<double>> x, y, z;
  for (double i = -5; i <= 5; i += 0.25) {
    std::vector<double> x_row, y_row, z_row;
    for (double j = -5; j <= 5; j += 0.25) {
      x_row.emplace_back(i);
      y_row.emplace_back(j);
      z_row.emplace_back(std::sin(std::hypot(i, j)));
    }
    x.emplace_back(x_row);
    y.emplace_back(y_row);
    z.emplace_back(z_row);
  }

  plt::plot_surface(x, y, z);
  plt::show();
}

void TestImshow() {
  const int ncols = 500, nrows = 300;
  std::vector<float> z(ncols * nrows);
  for (int j=0; j<nrows; ++j) {
    for (int i=0; i<ncols; ++i) {
      z.at(ncols * j + i) = std::sin(std::hypot(i - ncols/2, j - nrows/2));
    }
  }

  const float* zptr = &(z[0]);
  const int colors = 1;

  plt::title("My matrix");
  plt::imshow(zptr, nrows, ncols, colors);
  plt::show();
}

void TestImshowMnist() {
  auto dataset = torch::data::datasets::MNIST("/home/saminda/Data/mnist", torch::data::datasets::MNIST::Mode::kTest);
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      auto example = dataset.get(i*5 + j);
      plt::title(absl::Substitute("$0", example.target.item<int32_t>()));
      plt::subplot(5, 5, i*5+j+1);
      plt::imshow(example.data.data<float>(), example.data.size(1), example.data.size(2), /*colors=*/1);
    }
  }
  plt::show();
}

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);
  std::cout << "*** CIFAR ST ***" << std::endl;
//  TestMinimalExample();
//  TestComprehensiveExample();
//  Test3DFunction();
//  TestImshow();
  TestImshowMnist();
  std::cout << "*** CIFAR EN ***" << std::endl;
  return 0;
}
