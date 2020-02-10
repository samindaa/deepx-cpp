//
// Created by saminda on 1/30/20.
//

#include <iostream>
#include "spinup/algos/vpg/vpg.h"

void TestVpg() {
  auto seq = deepx::spinup::algos::vpg::mlp(/*sizes=*/{4, 3, 2}, /*activation=*/torch::tanh);
  std::cout << *seq->get() << std::endl;
}

int main(int argc, char **argv) {
  std::cout << "*** spinup start ***" << std::endl;
  TestVpg();
  std::cout << "*** spinup end   ***" << std::endl;
  return 0;
}

