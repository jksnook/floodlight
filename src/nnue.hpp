#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "incbin.h"
#include "types.hpp"

namespace Spotlight {

const int INPUT_SIZE = 768;
const int HIDDEN_SIZE = 64;
const int QA = 255;
const int QB = 64;
const int EVAL_SCALE = 400;

namespace NN {

struct Accumulator {
    void addPiece(Square from_sq, Square to_sq);

    void removePiece(Square from_sq, Square to_sq);

    void movePiece(Square from_sq, Square to_sq);

    std::array<int16_t, HIDDEN_SIZE> values;
};

inline int screlu(int input) { return std::pow(std::clamp(input, 0, QA), 2); }

int evaluate(Accumulator &acc);

void load();
};  // namespace NN

}  // namespace Spotlight
