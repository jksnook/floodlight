#include "nnue.hpp"

#include <immintrin.h>

#include <cassert>
#include <cstring>
#include <iostream>

#include "utils.hpp"

namespace Spotlight {

INCBIN(IncludedNetwork, "./src/net.bin");

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

#if defined(__AVX2__)

    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        __m256i dst_white = _mm256_loadu_si256((__m256i*)(&values[WHITE][i]));
        __m256i dst_black = _mm256_loadu_si256((__m256i*)(&values[BLACK][i]));

        __m256i add_white = _mm256_loadu_si256((__m256i*)(&hl_weights[white_index][i]));
        __m256i add_black = _mm256_loadu_si256((__m256i*)(&hl_weights[black_index][i]));

        dst_white = _mm256_add_epi16(dst_white, add_white);
        dst_black = _mm256_add_epi16(dst_black, add_black);

        _mm256_storeu_si256((__m256i*)(&values[WHITE][i]), dst_white);
        _mm256_storeu_si256((__m256i*)(&values[BLACK][i]), dst_black);
    }

#else

    for (int acc_idx = 0; acc_idx < HIDDEN_SIZE; acc_idx++) {
        values[WHITE][acc_idx] += hl_weights[white_index][acc_idx];
        values[BLACK][acc_idx] += hl_weights[black_index][acc_idx];
    }

#endif
}

void Accumulator::removePiece(Piece piece, Square sq) {
    int white_index = getIndex<WHITE>(piece, sq);
    int black_index = getIndex<BLACK>(piece, sq);

#if defined(__AVX2__)

    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        __m256i dst_white = _mm256_loadu_si256((__m256i*)(&values[WHITE][i]));
        __m256i dst_black = _mm256_loadu_si256((__m256i*)(&values[BLACK][i]));

        __m256i sub_white = _mm256_loadu_si256((__m256i*)(&hl_weights[white_index][i]));
        __m256i sub_black = _mm256_loadu_si256((__m256i*)(&hl_weights[black_index][i]));

        dst_white = _mm256_sub_epi16(dst_white, sub_white);
        dst_black = _mm256_sub_epi16(dst_black, sub_black);

        _mm256_storeu_si256((__m256i*)(&values[WHITE][i]), dst_white);
        _mm256_storeu_si256((__m256i*)(&values[BLACK][i]), dst_black);
    }

#else

    for (int acc_idx = 0; acc_idx < HIDDEN_SIZE; acc_idx++) {
        values[WHITE][acc_idx] -= hl_weights[white_index][acc_idx];
        values[BLACK][acc_idx] -= hl_weights[black_index][acc_idx];
    }

#endif
}

void Accumulator::movePiece(Piece piece, Square from_sq, Square to_sq) {
    int white_from_index = getIndex<WHITE>(piece, from_sq);
    int black_from_index = getIndex<BLACK>(piece, from_sq);

    int white_to_index = getIndex<WHITE>(piece, to_sq);
    int black_to_index = getIndex<BLACK>(piece, to_sq);

#if defined(__AVX2__)

    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        __m256i* dst_white_ptr = (__m256i*)(&values[WHITE][i]);
        __m256i* dst_black_ptr = (__m256i*)(&values[BLACK][i]);

        __m256i dst_white = _mm256_loadu_si256(dst_white_ptr);
        __m256i dst_black = _mm256_loadu_si256(dst_black_ptr);

        __m256i sub_white = _mm256_loadu_si256((__m256i*)(&hl_weights[white_from_index][i]));
        __m256i sub_black = _mm256_loadu_si256((__m256i*)(&hl_weights[black_from_index][i]));
        __m256i add_white = _mm256_loadu_si256((__m256i*)(&hl_weights[white_to_index][i]));
        __m256i add_black = _mm256_loadu_si256((__m256i*)(&hl_weights[black_to_index][i]));

        dst_white = _mm256_sub_epi16(dst_white, sub_white);
        dst_black = _mm256_sub_epi16(dst_black, sub_black);
        dst_white = _mm256_add_epi16(dst_white, add_white);
        dst_black = _mm256_add_epi16(dst_black, add_black);

        _mm256_storeu_si256(dst_white_ptr, dst_white);
        _mm256_storeu_si256(dst_black_ptr, dst_black);
    }

#else

    for (int acc_idx = 0; acc_idx < HIDDEN_SIZE; acc_idx++) {
        values[WHITE][acc_idx] -= hl_weights[white_from_index][acc_idx];
        values[BLACK][acc_idx] -= hl_weights[black_from_index][acc_idx];

        values[WHITE][acc_idx] += hl_weights[white_to_index][acc_idx];
        values[BLACK][acc_idx] += hl_weights[black_to_index][acc_idx];
    }

#endif
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

    // std::cout << hl_biases[0] << " " << hl_biases[1] << " " << output_bias << std::endl;
}

int evaluate(Accumulator& acc, Color side) {
    Color other_side = getOtherSide(side);

#if defined(__AVX2__)

    int32_t output = 0;

    const __m256i QA_vector = _mm256_set1_epi16(QA);

    const __m256i zeros = _mm256_set1_epi16(0);

    __m256i sum = zeros;

    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        // load values
        __m256i a = _mm256_loadu_si256((__m256i*)(&acc.values[side][i]));
        __m256i b = _mm256_loadu_si256((__m256i*)(&acc.values[other_side][i]));

        // load weights
        __m256i weights_a = _mm256_loadu_si256((__m256i*)(&output_weights[i]));
        __m256i weights_b = _mm256_loadu_si256((__m256i*)(&output_weights[HIDDEN_SIZE + i]));

        // apply crelu
        a = _mm256_min_epi16(a, QA_vector);
        a = _mm256_max_epi16(a, zeros);

        b = _mm256_min_epi16(b, QA_vector);
        b = _mm256_max_epi16(b, zeros);

        // apply weights
        __m256i a_w = _mm256_mullo_epi16(a, weights_a);
        __m256i b_w = _mm256_mullo_epi16(b, weights_b);

        // multiply again to get screlu and add to our vector sum
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(a, a_w));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(b, b_w));
    }

    sum = _mm256_hadd_epi32(sum, sum);
    sum = _mm256_hadd_epi32(sum, sum);

    output = _mm256_extract_epi32(sum, 0) + _mm256_extract_epi32(sum, 4);

    // std::cout << "raw eval: " << s << "\n";

    output += output_bias;

    output /= QA;

    output *= EVAL_SCALE;

    output /= QA * QB;

    // std::cout << output << "\n";

    return output;

#else

    int output = output_bias;

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        output += screlu(acc.values[side][i]) * output_weights[i];
        output += screlu(acc.values[other_side][i]) * output_weights[HIDDEN_SIZE + i];
    }

    output /= QA;

    output *= EVAL_SCALE;

    output /= QA * QB;

    return output;

#endif
}

}  // namespace NN

}  // namespace Spotlight