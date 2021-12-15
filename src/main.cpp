
#include <algorithm>
#include <iostream>
#include <random>
#include "timer.h"
#include "options.h"
#include "board.h"
#include "search.h"

std::mt19937_64 engine;
Options global_options;

auto random_board(size_t empties, std::mt19937_64& engine)
{
    while (true)
    {
        std::array<Square, 32> moves;
        Board board;
        Square prev_move = INVALID_MOVE;
        while (board.empties() > empties)
        {
            board.get_moves();
            board.get_moves_list(moves.data());
            auto num_moves = board.num_moves();
            if (num_moves == 0u)
            {
                board.move_empty();
                board.get_moves();
                board.get_moves_list(moves.data());
                num_moves = board.num_moves();
                if (num_moves == 0u)
                    break;
            }
            auto rand = std::uniform_int_distribution<size_t>(0, num_moves - 1)(engine);
            prev_move = moves[rand];
            board.move(moves[rand]);
        }
        if (board.empties() == empties)
            return std::make_pair(board, prev_move);
    }
}

auto generate_game(Searcher& searcher)
{
    constexpr float  value_not_available = -999.f;

    std::vector<Board> board_history;
    std::vector<Square> move_history;
    std::vector<float> value_history;
    std::vector<std::vector<std::pair<Square, float>>> policy_history;

    Board board;
    std::array<Square, 4> initial_moves{E6, F5, D3, C4};
    Square prev_move = initial_moves[std::uniform_int_distribution<int>(0, 3)(engine)];

    board_history.push_back(board);
    board.move(prev_move);
    move_history.push_back(prev_move);
    value_history.push_back(0.0f);
    policy_history.push_back({{E6, 1.0f}, {F5, 1.0f}, {D3, 1.0f}, {C4, 1.0f}});

    searcher.set_board(board, prev_move);

    while (!board.has_ended() && board.empties() > global_options.exact_empties)
    {
        board_history.push_back(board);
        //board.print();
        board.get_moves();

        // if there is only one choice
        if (board.num_moves() <= 1u)
        {
            //board.print();
            //std::cout << "only one choice.\n";
            if (board.num_moves() == 0u)
                prev_move = EMPTY_MOVE;
            else
                board.get_moves_list(&prev_move);
            board.move(prev_move);
            searcher.move(prev_move);
            move_history.push_back(prev_move);
            value_history.push_back(value_not_available);
            policy_history.push_back({{prev_move, 1.0f}});
            continue;
        }

        searcher.go(global_options.playouts);
        value_history.push_back(searcher.root().value_sum / float(searcher.root().visits));
        
        auto [visits, picked_move] = searcher.get_results();
        policy_history.push_back(visits);

        // make the move
        prev_move = picked_move;
        board.move(prev_move);
        searcher.move(prev_move);
        move_history.push_back(prev_move);
    }
    auto result = end_eval_with_empties(board, prev_move);
    if (global_options.print_moves)
    {
        for (auto move : move_history)
            std::cout << square_name(move) << " ";
        std::cout << " (last eval = " << result << ")\n";
    }

    auto estimated_value = -float(result);
    for (auto iter = value_history.rbegin(); iter != value_history.rend(); ++iter)
    {
        if (*iter == value_not_available)
            *iter = estimated_value;
        else
            estimated_value = *iter;
        estimated_value = -estimated_value;
    }
    return std::make_tuple(board_history, move_history, value_history, policy_history, result);
}

struct PositionBits
{
    uint64_t self_bits;
    uint64_t oppo_bits;
    uint64_t move_bits;
    uint8_t  policys[31]{};
    int8_t   result;
};

struct GameBits
{
    uint64_t num_positions;
    std::array<PositionBits, 120> positions;
};

template<typename Data>
auto gamedata_to_bits(const Data& data, GameBits& game_bits)
{
    const auto& [boards, moves, values, policys, result] = data;
    const auto game_length = boards.size();
    game_bits.num_positions = game_length;
    for (size_t i = 0; i < game_length; ++i)
    {
        auto board = boards[i];
        const auto& policy = policys[i];
        auto current_result = ((game_length - i) % 2) ? -result : result;
        auto current_value = values[i];
        auto current_q = 0.5f * (current_result + current_value);
        if (current_q < -1.0f || current_q > 1.0f)
            throw std::logic_error("");
        board.get_moves();
        game_bits.positions[i].self_bits = board.self_bits_;
        game_bits.positions[i].oppo_bits = board.oppo_bits_;
        game_bits.positions[i].move_bits = board.move_bits_;
        game_bits.positions[i].result = int8_t(std::round(100 * current_q));
        auto num_moves = board.num_moves();
        if (!((num_moves == 0u) ? (policy.size() == 1u) : (policy.size() == num_moves)))
        {
            board.print();
            std::cout << num_moves << "\n";
            std::cout << policy.size() << "\n";
            throw 0;
        }
        for (size_t j = 0; j < num_moves; ++j)
        {
            assert(0.0f <= policy[j].second && policy[j].second <= 1.0f);
            int val = int(std::floor(policy[j].second * 256.0f));
            val = std::max(std::min(val, 255), 0);
            game_bits.positions[i].policys[j] = uint8_t(val);
        }
    }
}

struct GameWriter
{
    FILE* str = nullptr;

    GameWriter(std::string file)
    {
        str = fopen(file.c_str(), "wb");
        if (!str)
            throw std::logic_error("");
    }

    ~GameWriter()
    {
        if (str)
            fclose(str);
        str = nullptr;
    }

    void write(const GameBits& bits)
    {
        fwrite((const void*)(&bits.num_positions), 8, 1, str);
        fwrite((const void*)(&bits.positions), sizeof(PositionBits), bits.num_positions, str);
    }
};

auto generate_games(Searcher& searcher, std::string file, size_t num_games)
{
    auto game_writer = GameWriter(file);
    for (size_t i = 0; i < num_games; ++i)
    {
        GameBits bits;
        auto game = generate_game(searcher);
        printf(".");
        gamedata_to_bits(game, bits);
        game_writer.write(bits);
    }
}

auto evaluate_new_net(Searcher& old_searcher, Searcher& new_searcher, size_t num_games)
{
    std::vector<int> results;
    Searcher* searcher_ptr = nullptr;

    for (size_t i_game = 0; i_game < num_games; ++i_game)
    {
        const bool old_net_moves_first = (i_game % 2) == 0;
        searcher_ptr = old_net_moves_first ? &new_searcher : &old_searcher;

        Board board;
        Square prev_move = INVALID_MOVE;

        while (!board.has_ended() && board.empties() > global_options.exact_empties)
        {
            if (searcher_ptr == &new_searcher)
                searcher_ptr = &old_searcher;
            else
                searcher_ptr = &new_searcher;
            
            board.get_moves();
            if (board.num_moves() <= 1u)
            {
                if (board.num_moves() == 0u)
                {
                    prev_move = EMPTY_MOVE;
                    board.move_empty();
                }
                else
                {
                    board.get_moves_list(&prev_move);
                    board.move(prev_move);
                }
                if (global_options.print_moves)
                    std::cout << square_name(prev_move) << " ";
                continue;
            }

            searcher_ptr->set_board(board, prev_move);
            searcher_ptr->go(global_options.playouts);
            auto [visits, picked_move] = searcher_ptr->get_results();
            prev_move = picked_move;
            board.move(prev_move);
            if (global_options.print_moves)
                std::cout << square_name(prev_move) << " ";
        }
        if (global_options.print_moves)
            std::cout << "\n";

        auto eval = end_eval_with_empties(board, prev_move);
        if (searcher_ptr == &new_searcher)
            eval = -eval;
        results.push_back(eval);
    }
    return results;
}

void interactive_session(Searcher& searcher)
{
    Board board;
    Square prev_move = INVALID_MOVE;
    searcher.set_board(board, prev_move);
    std::string command;

    while (true)
    {
        std::cout << "> ";
        std::getline (std::cin, command, '\n');
        if (command.substr(0, 8) == "setboard")
        {
            auto board_end = command.find(' ', 9);
            if (board_end == std::string::npos)
                continue;
            auto to_move = command[board_end + 1];
            board = Board(command.substr(9, board_end - 9), to_move == 'X' || to_move == 'b');
            prev_move = INVALID_MOVE;
            searcher.set_board(board, prev_move);
            board.print();
        }
        else if (command.substr(0, 8) == "textform")
        {
            std::cout << board.textform() << "\n";
        }
        else if (command.substr(0, 4) == "move")
        {
            auto move = string_to_move(command.substr(5, 2));
            if (move == INVALID_MOVE)
                continue;
            prev_move = move;
            board.move(prev_move);
            searcher.set_board(board, prev_move);
            board.print();
        }
        else if (command.substr(0, 2) == "go")
        {
            board.get_moves();
            if (board.num_moves() > 0)
            {
                searcher.go(global_options.playouts);
                auto [visits, picked_move] = searcher.get_results();
                prev_move = picked_move;
            }
            else
            {
                prev_move = EMPTY_MOVE;
            }
            board.move(prev_move);
            searcher.set_board(board, prev_move);
            std::cout << square_name(prev_move) << "\n";
        }
        else if (command.substr(0, 4) == "eval")
        {
            std::cout << end_eval_with_empties(board, INVALID_MOVE) << "\n";
        }
        else if (command.substr(0, 4) == "quit")
        {
            break;
        }
    }
}

int main(int argc, char* argv[])
{
    deploy_timer();
    std::random_device rd;
    engine.seed(rd());
    torch::NoGradGuard no_grad;

    parse_options((char**)argv, (char**)argv + argc);

    if (global_options.mode == Mode::Selfplay)
    {
        Evaluator<global_max_parallel, true> evaluator(global_options.net_path);
        Searcher searcher(evaluator);
        tick();
        generate_games(searcher, global_options.games_path, global_options.games);
        tock();
    }
    else if (global_options.mode == Mode::Compare)
    {
        Evaluator<global_max_parallel, true> old_net(global_options.alt_net_path);
        Evaluator<global_max_parallel, true> new_net(global_options.net_path);
        Searcher old_searcher(old_net);
        Searcher new_searcher(new_net);

        auto results = evaluate_new_net(old_searcher, new_searcher, global_options.games);
        for (auto result : results)
            std::cout << (result > 0 ? "w" : result < 0 ? "l" : "d");
        std::cout << "\n";
    }
    else if (global_options.mode == Mode::Interactive)
    {
        Evaluator<global_max_parallel, true> evaluator(global_options.net_path);
        Searcher searcher(evaluator);
        interactive_session(searcher);
    }
    else
    {
        std::cerr << "invalid mode\n";
    }
    return 0;
}
