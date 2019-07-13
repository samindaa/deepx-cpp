//
// Created by Saminda Abeyruwan on 2019-07-02.
//

#ifndef SAMINDA_AIMA_CPP__TILES3_H_
#define SAMINDA_AIMA_CPP__TILES3_H_

// http://incompleteideas.net/tiles/tiles3.py-remove

#include <cmath>
#include <vector>
#include <iostream>
#include <unordered_map>
#include "../abseil-cpp/absl/hash/hash.h"

class IHT {
 public:
  explicit IHT(const int &sizeval) : size_(sizeval), overfull_count_(0) {}

  int hashcoords(const std::vector<int> &coordinates) {
    if (dictionary_.find(coordinates) != dictionary_.end()) {
      return dictionary_[coordinates];
    }

    auto count = dictionary_.size();
    if (count >= size_) {
      if (overfull_count_ == 0) {
        std::cout << "IHT full, starting to allow collisions" << std::endl;
      }
      ++overfull_count_;
      return absl::Hash<std::vector<int>>{}(coordinates) % size_;
    } else {
      dictionary_[coordinates] = count;
      return count;
    }
  }

  friend std::ostream &operator<<(std::ostream &out, const IHT &src);

 private:
  int size_;
  int overfull_count_;
  std::unordered_map<std::vector<int>, int, absl::Hash<std::vector<int>>>
      dictionary_;

};

class TileCoder {
 public:
  explicit TileCoder(IHT &iht) : iht_(iht) {}

  // Returns num-tilings tile indices corresponding to the floats and ints
  std::vector<int> tiles(const int &numtilings,
                         const std::vector<float> &floats,
                         const std::vector<int> &ints = {}) {
    std::vector<float> qfloats;
    for (const auto &f : floats) {
      qfloats.emplace_back(std::floor(f * numtilings));
    }
    std::vector<int> tiles;
    for (int tiling = 0; tiling < numtilings; ++tiling) {
      const auto tilingX2 = tiling * 2;
      std::vector<int> coords{tiling};
      auto b = tiling;
      for (const auto q : qfloats) {
        coords.emplace_back(static_cast<int>(std::floor((q + b) / numtilings)));
        b += tilingX2;
      }
      coords.insert(coords.end(), ints.begin(), ints.end());
      tiles.emplace_back(iht_.hashcoords(coords));
    }
    return tiles;
  }

  std::vector<int> tileswrap(const int &numtilings,
                             const std::vector<float> &floats,
                             const std::vector<int> &wrapwidths,
                             const std::vector<int> &ints = {}) {
    std::vector<float> qfloats;
    for (const auto &f : floats) {
      qfloats.emplace_back(std::floor(f * numtilings));
    }
    std::vector<int> tiles;
    for (int tiling = 0; tiling < numtilings; ++tiling) {
      const auto tilingX2 = tiling * 2;
      std::vector<int> coords{tiling};
      auto b = tiling;
      for (int i = 0; i < qfloats.size(); ++i) {
        const auto q = qfloats[i];
        const auto width = wrapwidths[i];
        const auto
            c = static_cast<int>(std::floor((q + b % numtilings) / numtilings));
        coords.emplace_back(width != -1 ? c % width : c);
        b += tilingX2;
      }
      coords.insert(coords.end(), ints.begin(), ints.end());
      tiles.emplace_back(iht_.hashcoords(coords));
    }
    return tiles;
  }

 private:
  IHT &iht_;
};

std::ostream &operator<<(std::ostream &out, const IHT &src) {
  out << "Collision table:" <<
      "\nsize: " << src.size_ <<
      "\noverfullCount: " << src.overfull_count_ <<
      "\ndictionary: " << src.dictionary_.size() << " items" << std::endl;
  return out;
}

#endif //SAMINDA_AIMA_CPP__TILES3_H_
