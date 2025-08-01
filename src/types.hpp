#pragma once

#include <cstdint>

namespace Floodlight {

using U64 = std::uint64_t;
using BitBoard = std::uint64_t;
using move16 = std::uint16_t;

enum Color : uint8_t { WHITE, BLACK };

enum GenType : int { CAPTURES_AND_PROMOTIONS, QUIET, LEGAL };

// clang-format off
enum Piece : uint8_t {
    WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING,
    BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING,
    NO_PIECE
};

enum PieceType : uint8_t { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };


enum Square : uint16_t {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
};
// clang-format on

}  // namespace Floodlight
