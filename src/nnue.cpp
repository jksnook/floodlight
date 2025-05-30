#include "nnue.hpp"

#include <cstring>
#include <iostream>

namespace Spotlight {

INCBIN(IncludedNetwork, "./src/beans.bin");

NN::NN(): output_bias(0) {

}

void NN::load() {
    size_t mem_index = 0ULL;

    size_t hl_weights_bytes = INPUT_SIZE * HIDDEN_SIZE * sizeof(int16_t);
    std::memcpy(hl_weights.data(), &gIncludedNetworkData[0], INPUT_SIZE * HIDDEN_SIZE * sizeof(int16_t));
    mem_index = hl_weights_bytes;

    size_t hl_biases_bytes = HIDDEN_SIZE * sizeof(int32_t);
    std::memcpy(hl_biases.data(), &gIncludedNetworkData[mem_index], hl_biases_bytes);
    mem_index += hl_biases_bytes;

    size_t output_weights_bytes = HIDDEN_SIZE * sizeof(int16_t);
    std::memcpy(output_weights.data(), &gIncludedNetworkData[mem_index], output_weights_bytes);
    mem_index += output_weights_bytes;

    std::memcpy(&output_bias, &gIncludedNetworkData[mem_index], sizeof(int16_t));


    std::cout << hl_biases[0] << " " << hl_biases[1] << " " << output_bias << std::endl;
    
}


}