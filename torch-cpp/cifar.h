//
// Created by saminda on 7/19/19.
//

#ifndef SAMINDA_AIMA_CPP_TORCH_CPP_CIFAR_H_
#define SAMINDA_AIMA_CPP_TORCH_CPP_CIFAR_H_

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <c10/util/Exception.h>

namespace torch {
namespace data {
namespace datasets {

struct CIFAR : public Dataset<CIFAR> {
  /// The mode in which the dataset is loaded.
  enum class Mode { kTrain, kTest };

  /// Loads the CIFAR dataset from the `root` path.
  ///
  /// The supplied `root` path should contain the *content* of the unzipped
  /// CIFAR dataset, available from https://www.cs.toronto.edu/~kriz/cifar.html.
  explicit CIFAR(const std::string &root, Mode mode = Mode::kTrain);

  /// Returns the `Example` at the given `index`.
  Example<> get(size_t index) override;

  /// Returns the size of the dataset.
  optional<size_t> size() const override;

  /// Returns true if this is the training subset of CIFAR.
  bool is_train() const noexcept;

  /// Returns all images stacked into a single tensor.
  const Tensor &images() const;

  /// Returns all targets stacked into a single tensor.
  const Tensor &targets() const;

 private:
  bool train_;
  Tensor images_, targets_;
};
} // namespace datasets
} // namespace data
} // namespace torch



#endif //SAMINDA_AIMA_CPP_TORCH_CPP_CIFAR_H_
