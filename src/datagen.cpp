#include "datagen.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace Floodlight {

DataEntry::DataEntry(Position& pos, int _score, int _result) {
    for (auto& p : pieces) {
        p = 0;
    }

    if (pos.side_to_move == WHITE) {
        result = _result;
        score = _score;

        occupancy = pos.bitboards[OCCUPANCY];
        BitBoard temp = occupancy;
        int pieces_idx = 0;

        while (temp) {
            Square sq = popLSB(temp);

            Piece p = pos.at(sq);
            uint8_t side = p >= 6;

            uint8_t bullet_piece = (side << 3) | (p % 6);

            pieces[pieces_idx / 2] |= bullet_piece << ((pieces_idx & 1) * 4);
            pieces_idx++;
        }

        ksq = bitScanForward(pos.bitboards[getPieceID(KING, pos.side_to_move)]);
        opp_ksq = bitScanForward(pos.bitboards[getPieceID(KING, getOtherSide(pos.side_to_move))]);

    } else {
        occupancy = 0ULL;
        int pieces_idx = 0;

        result = 2 - _result;
        score = -_score;

        for (int rank = 7; rank >= 0; rank--) {
            for (int file = 0; file < 8; file++) {
                Square sq = static_cast<Square>(rank * 8 + file);
                Piece p = pos.at(sq);

                if (p != NO_PIECE) {
                    assert(pieces_idx < 32);
                    Piece p_relative = FLIP_PIECE[p];
                    uint8_t side = p_relative >= 6;
                    uint8_t bullet_piece = (side << 3) | (p_relative % 6);
                    pieces[pieces_idx / 2] = pieces[pieces_idx / 2] |= bullet_piece
                                                                       << ((pieces_idx & 1) * 4);
                    ;
                    occupancy |= setBit(static_cast<int>(sq) ^ 56);
                    pieces_idx++;
                }
            }
        }

        ksq = bitScanForward(pos.bitboards[BLACK_KING]) ^ 56;
        opp_ksq = bitScanForward(pos.bitboards[WHITE_KING]) ^ 56;
    }

    side = pos.side_to_move;

    for (auto& e : extra) {
        e = 0;
    }
}

void DataEntry::print() {
    std::cout << "occupancy: \n";
    printBitboard(occupancy);
    std::cout << "score: " << static_cast<int>(score) << "\n";
    std::cout << "result: " << static_cast<int>(result) << "\n";
    std::cout << "ksq: " << static_cast<int>(ksq) << "\nopp_ksq: " << static_cast<int>(opp_ksq)
              << "\n";
    std::cout << "side: " << static_cast<int>(side) << "\n";

    int piece_idx = 0;

    std::array<Piece, 64> board;

    BitBoard temp = occupancy;

    while (temp) {
        board[popLSB(temp)] =
            static_cast<Piece>((pieces[piece_idx / 2] >> ((piece_idx & 1) * 4)) & 0b1111);
        piece_idx++;
    }

    std::cout << "+---+---+---+---+---+---+---+---+\n";
    for (int rank = 7; rank >= 0; rank--) {
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file;
            if (setBit(sq) & occupancy) {
                Piece piece = board[sq];
                std::cout << "| " << PIECE_TO_LETTER_MAP[piece] << ' ';
                piece_idx++;
            } else {
                std::cout << "|   ";
            }
        }
        std::cout << "|\n+---+---+---+---+---+---+---+---+\n";
    }
}

bool isQuiet(MoveList& moves) {
    for (const auto& m : moves) {
        if (getMoveType(m.move) & CAPTURE_MOVE) {
            return false;
        }
    }
    return true;
}

void selfplay(int num_games, int num_threads, U64 node_count) {
    std::vector<std::thread> threads;
    std::mutex mx;
    int games_played = 0;

    std::ofstream output_file;
    output_file.open("./selfplay", std::ios::out | std::ios::binary);
    if (!output_file.is_open()) return;

    for (int i = 0; i < num_threads; i++) {
        threads.push_back(std::thread(playGames, num_games, node_count, i, std::ref(games_played),
                                      std::ref(output_file), std::ref(mx)));
    }

    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }

    output_file.close();
}

void playGames(int num_games, U64 node_count, int id, int& games_played, std::ofstream& output_file,
               std::mutex& mx) {
    Position pos;

    TT tt;
    std::atomic<bool> is_stopped(false);

    Search search(&tt, &is_stopped, [&]() { return search.nodes_searched; });
    search.make_output = false;

    std::random_device r;
    std::mt19937 myRandom(r());

    for (int i = 0; i < num_games; i++) {
        mx.lock();
        games_played++;
        if (games_played > num_games) {
            mx.unlock();
            break;
        };
        std::cout << "starting game " << games_played << " of " << num_games << " on thread " << id
                  << " with the following position and number of random moves\n";
        mx.unlock();
        pos = Position();
        tt.clear();
        search.clearHistory();

        std::vector<DataEntry> entries;

        int result = 0;

        int num_random =
            (myRandom() % (MAX_RANDOM_MOVES - MIN_RANDOM_MOVES + 1)) + MIN_RANDOM_MOVES;
        std::cout << num_random << "\n";

        // make some random moves
        for (int i = 0; i < num_random; i++) {
            // check for draw by repetition
            if (pos.isTripleRepetition()) {
                std::cout << "Draw" << std::endl;
                result = 1;
                break;
            }

            MoveList moves;
            generateMoves(moves, pos);

            // check for checkmate or stalemate
            if (moves.size() == 0) {
                if (inCheck(pos)) {
                    if (pos.side_to_move == WHITE) {
                        std::cout << "Black wins" << std::endl;
                        result = 0;
                    } else {
                        std::cout << "White wins" << std::endl;
                        result = 2;
                    }
                } else {
                    std::cout << "Stalemate" << std::endl;
                    result = 1;
                }
                break;
            }

            int random_index = myRandom() % moves.size();

            pos.makeMove(moves[random_index].move);
        }
        // pos.print();

        // main game loop
        while (true) {
            // Check for draw conditions
            if (pos.fifty_move >= FIFTY_MOVE_LIMIT) {
                std::cout << "Draw by fifty moves rule\n";
                // pos.print();
                result = 1;
                break;
            } else if (pos.isTripleRepetition()) {
                std::cout << "Draw by triple repetition\n";
                // pos.print();
                result = 1;
                break;
            } else if (countBits(pos.bitboards[OCCUPANCY]) == 2) {
                std::cout << "Draw by insufficient material\n";
                // pos.print();
                result = 1;
                break;
            }

            MoveList moves;
            generateMoves(moves, pos);

            // check for checkmate or stalemate
            if (moves.size() == 0) {
                if (inCheck(pos)) {
                    if (pos.side_to_move == WHITE) {
                        std::cout << "Black wins" << std::endl;
                        result = 0;
                    } else {
                        std::cout << "White wins" << std::endl;
                        result = 2;
                    }
                } else {
                    std::cout << "Stalemate" << std::endl;
                    result = 1;
                }
                break;
            }

            is_stopped.store(false);
            SearchResult search_result =
                search.nodeSearch(pos, MAX_PLY, node_count + myRandom() % (node_count / 10));

            move16 move = search_result.move;
            int score = search_result.score;

            int eval_score = eval(pos);
            int qscore = search.qScore(pos);

            if (eval_score == qscore && score < MATE_THRESHOLD && score > -MATE_THRESHOLD) {
                entries.emplace_back(pos, eval_score, result);
            }

            pos.makeMove(move);
            tt.nextGeneration();
        }

        mx.lock();
        if (output_file.is_open()) {
            // log positions to the output file
            std::cout << "positions written: " << entries.size() << "\n";
            for (auto& entry : entries) {
                if (entry.side == WHITE) {
                    entry.result = result;
                } else {
                    entry.result = 2 - result;
                    entry.side = WHITE;
                }
                output_file.write(reinterpret_cast<char*>(&entry), sizeof(DataEntry));
            }
        }
        mx.unlock();
    }
}

}  // namespace Floodlight
