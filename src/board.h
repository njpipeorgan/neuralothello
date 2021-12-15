#pragma once
#include <immintrin.h>
#include <cstdint>
#include <array>
#include <iostream>
#include <string>

#include "dir_mask.h"

bool _BitScanForward64(unsigned int* index, uint64_t mask)
{
    *index = __builtin_ffsll(mask) - 1;
    return mask;
}

template<ptrdiff_t Direction>
void shift_bitboard_impl(uint64_t& bits)
{
    static_assert(
        Direction == -9 || Direction == -8 || Direction == -7 ||
        Direction == -1 || Direction == +1 || Direction == +7 ||
        Direction == +8 || Direction == +9);
    if constexpr (Direction == -9)
        bits = (bits & 0xfefefefefefefefeu) >> 9;
    else if constexpr (Direction == -8)
        bits = bits >> 8;
    else if constexpr (Direction == -7)
        bits = (bits & 0x7f7f7f7f7f7f7f7fu) >> 7;
    else if constexpr (Direction == -1)
        bits = (bits & 0xfefefefefefefefeu) >> 1;
    else if constexpr (Direction == +1)
        bits = (bits & 0x7f7f7f7f7f7f7f7fu) << 1;
    else if constexpr (Direction == +7)
        bits = (bits & 0xfefefefefefefefeu) << 7;
    else if constexpr (Direction == +8)
        bits = bits << 8;
    else if constexpr (Direction == +9)
        bits = (bits & 0x7f7f7f7f7f7f7f7fu) << 9;
}

template<ptrdiff_t Direction, typename... Bits>
void shift_bitboard(Bits&... bits)
{
    (shift_bitboard_impl<Direction>(bits), ...);
}

inline void shift_bitboard_avx2(__m256i& bits_left, __m256i& bits_right)
{
    const __m256i shift_left  = _mm256_set_epi64x(9, 8, 7, 1);
    const __m256i shift_right = _mm256_set_epi64x(1, 7, 8, 9);
    const __m256i mask_left   = _mm256_set_epi64x(0x7f7f7f7f7f7f7f7fu,
        0xffffffffffffffffu, 0xfefefefefefefefeu, 0x7f7f7f7f7f7f7f7fu);
    const __m256i mask_right  = _mm256_set_epi64x(0xfefefefefefefefeu,
        0x7f7f7f7f7f7f7f7fu, 0xffffffffffffffffu, 0xfefefefefefefefeu);
    bits_left  = _mm256_and_si256(bits_left, mask_left);
    bits_right = _mm256_and_si256(bits_right, mask_right);
    bits_left  = _mm256_sllv_epi64(bits_left, shift_left);
    bits_right = _mm256_srlv_epi64(bits_right, shift_right);
}

using DirMoves = std::array<uint64_t, 8>;

enum Square : uint8_t
{
    H8, G8, F8, E8, D8, C8, B8, A8,
    H7, G7, F7, E7, D7, C7, B7, A7,
    H6, G6, F6, E6, D6, C6, B6, A6,
    H5, G5, F5, E5, D5, C5, B5, A5,
    H4, G4, F4, E4, D4, C4, B4, A4,
    H3, G3, F3, E3, D3, C3, B3, A3,
    H2, G2, F2, E2, D2, C2, B2, A2,
    H1, G1, F1, E1, D1, C1, B1, A1,
    EMPTY_MOVE, INVALID_MOVE
};

inline std::string square_name(Square square)
{
    if (square == EMPTY_MOVE)
        return std::string("--");
    else if (square >= INVALID_MOVE)
        return std::string("??");
    else
    {
        char name[3] = "a1";
        name[0] += 7 - (int(square) % 8);
        name[1] += 7 - (int(square) / 8);
        return std::string(name);
    }
}

struct Board
{
    uint64_t self_bits_ = make_bitboard(D5, E4);
    uint64_t oppo_bits_ = make_bitboard(D4, E5);

    mutable uint64_t move_bits_ = 0u;
    bool black_to_move_ = true;

    Board() = default;

    Board(uint64_t self_bits, uint64_t oppo_bits, bool black_to_move = true) :
        self_bits_{self_bits}, oppo_bits_{oppo_bits},
        black_to_move_{black_to_move}
    {
    }

    Board(const std::string& board, bool black_to_move) :
        self_bits_{}, oppo_bits_{}, black_to_move_{black_to_move}
    {
        uint64_t* black_ptr = black_to_move ? &self_bits_ : &oppo_bits_;
        uint64_t* white_ptr = black_to_move ? &oppo_bits_ : &self_bits_;
        for (size_t i = 0; i < 64 && i < board.length(); ++i)
        {
            switch (board[i])
            {
            case 'X': case 'x': case 'B': case 'b':
                *black_ptr |= uint64_t(1) << (63 - i); break;
            case 'O': case 'o': case 'W': case 'w':
                *white_ptr |= uint64_t(1) << (63 - i); break;
            }
        }
    }

    std::string textform() const
    {
        std::string str(66, ' ');
        const auto* black_ptr = black_to_move_ ? &self_bits_ : &oppo_bits_;
        const auto* white_ptr = black_to_move_ ? &oppo_bits_ : &self_bits_;
        for (size_t i = 0; i < 64; ++i)
        {
            if ((*black_ptr) & (uint64_t(1) << (63 - i)))
                str[i] = 'X';
            else if ((*white_ptr) & (uint64_t(1) << (63 - i)))
                str[i] = 'O';
            else 
                str[i] = '-';
        }
        str[65] = black_to_move_ ? 'X' : 'O';
        return str;
    }

    template<typename... Squares>
    static uint64_t make_bitboard(Squares... s)
    {
        return ((uint64_t(1) << s) | ...);
    }

    static uint64_t make_bitboard()
    {
        return 0;
    }

    DirMoves get_moves() const
    {
        auto move_bits_left  = _mm256_setzero_si256();
        auto move_bits_right = move_bits_left;
        auto self_bits_left  = _mm256_set1_epi64x(self_bits_);
        auto self_bits_right = self_bits_left;
        auto oppo_bits_left  = _mm256_set1_epi64x(oppo_bits_);
        auto oppo_bits_right = oppo_bits_left;
        auto mask_bits_left  = _mm256_set1_epi64x(~(self_bits_ | oppo_bits_));
        auto mask_bits_right = mask_bits_left;
        shift_bitboard_avx2(self_bits_left, self_bits_right);
        shift_bitboard_avx2(oppo_bits_left, oppo_bits_right);
        mask_bits_left  = _mm256_and_si256(mask_bits_left, oppo_bits_left);
        mask_bits_right = _mm256_and_si256(mask_bits_right, oppo_bits_right);
        for (size_t i = 1; i <= 6; ++i)
        {
            shift_bitboard_avx2(self_bits_left, self_bits_right);                                                     
            shift_bitboard_avx2(oppo_bits_left, oppo_bits_right);
            move_bits_left  = _mm256_or_si256(move_bits_left, _mm256_and_si256(mask_bits_left, self_bits_left));
            move_bits_right = _mm256_or_si256(move_bits_right, _mm256_and_si256(mask_bits_right, self_bits_right));
            mask_bits_left  = _mm256_and_si256(mask_bits_left, oppo_bits_left);
            mask_bits_right = _mm256_and_si256(mask_bits_right, oppo_bits_right);
        }

        DirMoves dir_moves;
        _mm256_storeu_si256((__m256i*)(&dir_moves[4]), move_bits_left);
        _mm256_storeu_si256((__m256i*)(&dir_moves[0]), move_bits_right);
        move_bits_left = _mm256_or_si256(move_bits_left, move_bits_right);
        auto move_bits_128 = _mm_or_si128(_mm256_castsi256_si128(move_bits_left),
            _mm256_extractf128_si256(move_bits_left, 1));
        move_bits_ = uint64_t(_mm_extract_epi64(move_bits_128, 0) | _mm_extract_epi64(move_bits_128, 1));
        return dir_moves;
    }

    bool has_ended() const
    {
        if (empties() == 0u)
            return true;
        auto copy = *this;
        copy.get_moves();
        if (copy.num_moves() > 0u)
            return false;
        copy.move_empty();
        copy.get_moves();
        if (copy.num_moves() > 0u)
            return false;
        return true;
    }

    size_t num_moves() const
    {
        return _mm_popcnt_u64(move_bits_);
    }

    template<typename Integral>
    void get_moves_list(Integral* moves) const
    {
        auto move_bits = move_bits_;
        unsigned int square;
        while (_BitScanForward64(&square, move_bits))
        {
            move_bits &= move_bits - 1u;
            *moves++ = Integral(square);
        }
    }

    void move(uint64_t square)
    {
        auto dir_moves = get_moves();
        if (!move_bits_)
        {
            if (square == EMPTY_MOVE)
                move_empty();
            else
                throw std::logic_error("");
        }
        else
        {
            if ((uint64_t(1) << square) & move_bits_)
                move(square, dir_moves);
            else
                throw std::logic_error("");
        }
    }

    void move(uint64_t square, const DirMoves& dir_moves)
    {
        auto move_mask = uint64_t(1) << square;
        auto* dir_mask_base = dir_mask[square];
        uint64_t all_flip_mask = move_mask;
        for (size_t i = 0; i < 4; ++i)
        {
            if (move_mask & dir_moves[i])
            {
                auto flip_mask = _pext_u64(self_bits_, dir_mask_base[i]);
                flip_mask = (flip_mask - 1u) & ~flip_mask;
                all_flip_mask |= _pdep_u64(flip_mask, dir_mask_base[i]);
            }
        }
        for (size_t i = 4; i < 8; ++i)
        {
            if (move_mask & dir_moves[i])
            {
                auto flip_mask = _pext_u64(self_bits_, dir_mask_base[i]);
                flip_mask = uint64_t(0xffffffff00000000u) >>
                    (_lzcnt_u32(uint32_t(flip_mask)) % 32u);
                all_flip_mask |= _pdep_u64(flip_mask, dir_mask_base[i]);
            }
        }
        self_bits_ |= all_flip_mask;
        oppo_bits_ &= ~all_flip_mask;
        move_empty();
    }

    void move_empty()
    {
        black_to_move_ = !black_to_move_;
        std::swap(self_bits_, oppo_bits_);
    }

    template<bool MakeMove>
    bool _is_valid_move_impl(Square square)
    {
        auto move_mask = uint64_t(1) << square;
        if (move_mask & (self_bits_ | oppo_bits_))
            return false;
        auto* dir_mask_base = dir_mask[square];
        uint64_t all_flip_mask = move_mask;

        for (size_t i = 0; i < 4; ++i)
        {
            auto flip_mask = _pext_u64(self_bits_, dir_mask_base[i]);
            flip_mask = (flip_mask - 1u) &
                (~flip_mask & (0u - uint64_t(flip_mask != 0u)));
            flip_mask = _pdep_u64(flip_mask, dir_mask_base[i]);
            bool is_valid = flip_mask && !(flip_mask & ~oppo_bits_);
            if (is_valid)
            {
                if constexpr (MakeMove)
                    all_flip_mask |= flip_mask;
                else
                    return true;
            }
        }
        for (size_t i = 4; i < 8; ++i)
        {
            auto flip_mask = _pext_u64(self_bits_, dir_mask_base[i]);
            flip_mask = uint64_t(0xffffffff00000000u) >>
                (_lzcnt_u32(uint32_t(flip_mask)) % 32u);
            flip_mask = _pdep_u64(flip_mask, dir_mask_base[i]);
            bool is_valid = flip_mask && !(flip_mask & ~oppo_bits_);
            if (is_valid)
            {
                if constexpr (MakeMove)
                    all_flip_mask |= flip_mask;
                else
                    return true;
            }
        }
        if constexpr (!MakeMove)
        {
            return false;
        }
        else if (all_flip_mask != move_mask)
        {
            self_bits_ |= all_flip_mask;
            oppo_bits_ &= ~all_flip_mask;
            move_empty();
            return true;
        }
        else
        {
            return false;
        }
    }

    bool is_valid_move(Square square)
    {
        return _is_valid_move_impl<false>(square);
    }

    bool move_if_valid(Square square)
    {
        return _is_valid_move_impl<true>(square);
    }

    size_t empties() const
    {
        return _mm_popcnt_u64(~(self_bits_ | oppo_bits_));
    }

    void print() const
    {
        std::string table[] =
        {
            "     a b c d e f g h   ",
            "   -------------------  ",
            " 1 |                 | 1",
            " 2 |   .         .   | 2",
            " 3 |                 | 3",
            " 4 |                 | 4",
            " 5 |                 | 5",
            " 6 |                 | 6",
            " 7 |   .         .   | 7",
            " 8 |                 | 8",
            "   -------------------  ",
            "     a b c d e f g h    "
        };

        auto black_bits = black_to_move_ ? self_bits_ : oppo_bits_;
        auto white_bits = black_to_move_ ? oppo_bits_ : self_bits_;
        get_moves();

        for (size_t file = 0; file < 8; ++file)
        {
            for (size_t rank = 0; rank < 8; ++rank)
            {
                auto* pixel = &table[(7 - file) + 2][2 * (7 - rank) + 5];
                uint64_t mask = uint64_t(1) << (file * 8 + rank);
                if ((black_bits & mask) && !(white_bits & mask))
                    *pixel = 'X';
                else if ((white_bits & mask) && !(black_bits & mask))
                    *pixel = 'O';
                else if ((black_bits & mask) && (white_bits & mask))
                    *pixel = '?';
                if (move_bits_ & mask)
                {
                    if (*pixel == 'X' || *pixel == 'O' || *pixel == '?')
                        *(pixel + 1) = '*';
                    else
                        *pixel = '*';
                }
            }
        }
        for (auto& rows : table)
            std::puts(rows.c_str());
    }
};

inline Square string_to_move(const std::string& str)
{
    if (str.length() != 2)
        return INVALID_MOVE;
    else if (str == "--")
        return EMPTY_MOVE;
    else if (str == "PS")
        return EMPTY_MOVE;
    else if ('a' <= str[0] && str[0] <= 'h' && '1' <= str[1] && str[1] <= '8')
        return Square(('8' - str[1]) * 8 + ('h' - str[0]));
    else if ('A' <= str[0] && str[0] <= 'H' && '1' <= str[1] && str[1] <= '8')
        return Square(('8' - str[1]) * 8 + ('H' - str[0]));
    else
        return INVALID_MOVE;
}
