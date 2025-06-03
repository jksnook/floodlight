#include "nnue.hpp"

#include <cstring>
#include <iostream>
#include <cassert>

#include "utils.hpp"

namespace Spotlight {

INCBIN(IncludedNetwork, "./src/beans.bin");

namespace NN {

std::array<std::array<int16_t, HIDDEN_SIZE>, INPUT_SIZE> hl_weights;

std::array<int16_t, HIDDEN_SIZE> hl_biases;

std::array<int16_t, HIDDEN_SIZE * 2> output_weights;

int16_t output_bias = 0;

void Accumulator::clear() {

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        values[WHITE][i] = hl_biases[i];
        values[BLACK][i] = hl_biases[i];
    }

}

void Accumulator::addPiece(Piece piece, Square sq) {

    int white_index = getIndex<WHITE>(piece, sq);
    int black_index = getIndex<BLACK>(piece, sq);

    assert(white_index < HIDDEN_SIZE * INPUT_SIZE);
    assert(black_index < HIDDEN_SIZE * INPUT_SIZE);


    for (int acc_idx = 0; acc_idx < HIDDEN_SIZE; acc_idx++) {
        values[WHITE][acc_idx] += hl_weights[white_index][acc_idx];
        values[BLACK][acc_idx] += hl_weights[black_index][acc_idx];
    }

}

void Accumulator::removePiece(Piece piece, Square sq) {
    int white_index = getIndex<WHITE>(piece, sq);
    int black_index = getIndex<BLACK>(piece, sq);


    for (int acc_idx = 0; acc_idx < HIDDEN_SIZE; acc_idx++) {
        values[WHITE][acc_idx] -= hl_weights[white_index][acc_idx];
        values[BLACK][acc_idx] -= hl_weights[black_index][acc_idx];
    }
}

void Accumulator::movePiece(Piece piece, Square from_sq, Square to_sq) {
    int white_from_index = getIndex<WHITE>(piece, from_sq);
    int black_from_index = getIndex<BLACK>(piece, from_sq);

    int white_to_index = getIndex<WHITE>(piece, to_sq);
    int black_to_index = getIndex<BLACK>(piece, to_sq);


    for (int acc_idx = 0; acc_idx < HIDDEN_SIZE; acc_idx++) {
        values[WHITE][acc_idx] -= hl_weights[white_from_index][acc_idx];
        values[BLACK][acc_idx] -= hl_weights[black_from_index][acc_idx];

        values[WHITE][acc_idx] += hl_weights[white_to_index][acc_idx];
        values[BLACK][acc_idx] += hl_weights[black_to_index][acc_idx];
    }
}

void load() {
    size_t mem_index = 0ULL;

    size_t hl_weights_bytes = INPUT_SIZE * HIDDEN_SIZE * sizeof(int16_t);
    std::memcpy(hl_weights.data(), &gIncludedNetworkData[0],
                INPUT_SIZE * HIDDEN_SIZE * sizeof(int16_t));
    mem_index = hl_weights_bytes;

    size_t hl_biases_bytes = HIDDEN_SIZE * sizeof(int16_t);
    std::memcpy(hl_biases.data(), &gIncludedNetworkData[mem_index], hl_biases_bytes);
    mem_index += hl_biases_bytes;

    size_t output_weights_bytes = HIDDEN_SIZE * 2 * sizeof(int16_t);
    std::memcpy(output_weights.data(), &gIncludedNetworkData[mem_index], output_weights_bytes);
    mem_index += output_weights_bytes;

    std::memcpy(&output_bias, &gIncludedNetworkData[mem_index], sizeof(int16_t));

    std::cout << hl_biases[0] << " " << hl_biases[1] << " " << output_bias << std::endl;
}

int evaluate(Accumulator &acc, Color side) {

    int32_t output = 0;

    Color other_side = getOtherSide(side);

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        output += screlu(acc.values[side][i]) * output_weights[i] / QA;
        output += screlu(acc.values[other_side][i]) * output_weights[HIDDEN_SIZE + i] / QA;
    }

    std::cout << output << "\n";

    output += output_bias;

    output *= EVAL_SCALE;

    output /= QA * QB;

    std::cout << output << "\n";

    return output;
}

}  // namespace NN

}  // namespace Spotlight