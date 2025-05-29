#pragma once

#include <algorithm>
#include <cmath>
#include <array>

#include "incbin.h"

namespace Spotlight {

const int HIDDEN_SIZE = 512;
const int QA = 255;
const int QB = 64;
const int EVAL_SCALE = 400;

INCBIN(NN_DATA, "beans.bin");

using Accumulator = std::array<std::array<int, 512>, 2>;

class Network {
public:

  inline int screlu(int input) { return std::pow(std::clamp(input, 0, QA), 2);}

  int evaluate(Accumulator &acc);

private:

  std::array<std::array<int, 768>, 512> hl_weights;

  std::array<int, 512> hl_biases;

  std::array<int, 512> output_weights;

  int output_bias;
};

}
