#include "datagen.hpp"
#include "eval.hpp"
#include "move.hpp"
#include "nnue.hpp"
#include "position.hpp"
#include "test.hpp"
#include "tuner.hpp"
#include "types.hpp"
#include "uci.hpp"
#include "zobrist.hpp"

using namespace Spotlight;

int main(int argc, char* argv[]) {
    initMoves();
    initMagics();
    initZobrist();

    NN::load();

    // Position pos;

    // NN::Accumulator acc;

    // pos.refreshAcc(acc);

    // // for (int i = 0; i < 16; i++) {
    // //     std::cout << acc.values[WHITE][i] << "\n";
    // //     std::cout << acc.values[BLACK][i] << "\n";
    // // }

    // NN::evaluate(acc, WHITE);

    // pos.readFen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ");

    // std::cout << "kiwipete: \n";

    // pos.refreshAcc(acc);

    // // for (int i = 0; i < 16; i++) {
    // //     std::cout << acc.values[WHITE][i] << "\n";
    // //     std::cout << acc.values[BLACK][i] << "\n";
    // // }

    // NN::evaluate(acc, WHITE);

    if (argc == 1) {
        UCI uci;
        uci.loop();
    } else if (static_cast<std::string>(argv[1]) == "searchtest") {
        testSearch();
    } else if (static_cast<std::string>(argv[1]) == "fulltest") {
        runTests();
    } else if (static_cast<std::string>(argv[1]) == "tune") {
        Tuner tuner;

        tuner.run();
        tuner.printWeights();
        tuner.outputToFile();
    } else if (static_cast<std::string>(argv[1]) == "datagen") {
        if (argc == 5) {
            selfplay(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]));
        } else {
            selfplay(100, NUM_THREADS, BASE_NODE_COUNT);
        }
    }
}