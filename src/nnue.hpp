#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "incbin.h"
#include "types.hpp"
#include "utils.hpp"

namespace Spotlight {

const int INPUT_SIZE = 768;
const int HIDDEN_SIZE = 64;
const int QA = 255;
const int QB = 64;
const int EVAL_SCALE = 400;

namespace NN {

template <Color side>
int getIndex(Piece piece, Square sq) {
    if constexpr(side == WHITE) {
        return piece * 64 + sq;
    } else {
        return (getPieceType(piece) + 6 * getOtherSide(getPieceColor(piece))) * 64 + (sq ^ 56);
    }
}


struct Accumulator {
    void clear();

    void addPiece(Piece piece, Square sq);

    void removePiece(Piece piece, Square sq);

    void movePiece(Piece piece, Square from_sq, Square to_sq);

    std::array<std::array<int16_t, HIDDEN_SIZE>, 2> values;
};

void load();

inline int screlu(int input) { return std::pow(std::clamp(input, 0, QA), 2); }

int evaluate(Accumulator &acc);

};  // namespace NN

}  // namespace Spotlight
