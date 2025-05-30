#pragma once

#include <algorithm>
#include <cmath>
#include <array>

#include "incbin.h"

namespace Spotlight {

const int INPUT_SIZE = 768;
const int HIDDEN_SIZE = 64;
const int QA = 255;
const int QB = 64;
const int EVAL_SCALE = 400;

// const size_t total_bytes = (INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE) * sizeof(int16_t) + (HIDDEN_SIZE + 1)* sizeof(int32_t);

using Accumulator = std::array<std::array<int, 512>, 2>;

class NN {
public:

  NN();

  inline int screlu(int input) { return std::pow(std::clamp(input, 0, QA), 2);}

  int evaluate(Accumulator &acc);

  void load();

private:

  std::array<std::array<int16_t, INPUT_SIZE>, HIDDEN_SIZE> hl_weights;

  std::array<int16_t, HIDDEN_SIZE> hl_biases;

  std::array<int32_t, HIDDEN_SIZE> output_weights;

  int16_t output_bias;
};

}
