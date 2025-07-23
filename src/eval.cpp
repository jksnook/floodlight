#include "eval.hpp"

#include "bitboards.hpp"

namespace Floodlight {

int eval(Position &pos) {
    return NN::evaluate(pos.accumulators.back(), pos.side_to_move);
}

}  // namespace Floodlight
