#pragma once

#include <fstream>
#include <mutex>
#include <string>

#include "position.hpp"
#include "search.hpp"

namespace Spotlight {

const int NUM_THREADS = 12;
const int FIFTY_MOVE_LIMIT = 20;
const int MAX_RANDOM_MOVES = 15;
const int MIN_RANDOM_MOVES = 5;
const int BASE_NODE_COUNT = 5000;

enum OutputType { FEN, BULLET };

inline std::string resultToString(int result) {
    switch (result) {
        case 0:
            return "0";
            break;
        case 1:
            return "0.5";
            break;
        case 2:
            return "1";
            break;
        default:
            break;
    }
    return "";
}

// data entry format based on BulletFormat
struct DataEntry {
    BitBoard occupancy;
    uint8_t pieces[16];
    int16_t score;
    uint8_t result;
    uint8_t ksq;
    uint8_t opp_ksq;
    uint8_t side;
    uint8_t extra[2];

    DataEntry(){};

    DataEntry(Position &pos, int _score, int _result);

    void print();
};

const size_t DATA_ENTRY_SIZE = sizeof(DataEntry);

bool isQuiet(MoveList &moves);

void selfplay(int num_games, int num_threads, U64 node_count);

void playGames(int num_games, U64 node_count, int id, int &games_played, std::ofstream &output_file,
               std::mutex &mx);

}  // namespace Spotlight
