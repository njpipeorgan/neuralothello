#pragma once
#include <algorithm>
#include <array>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <future>
#include <optional>
#include <variant>
#include <vector>

#include "options.h"
#include "board.h"
#include "nn.h"

constexpr float MIN_SCORE = -99.0;
constexpr float MAX_SCORE = 99.0;

extern std::mt19937_64 engine;

struct SearchStack
{
    static constexpr size_t max_num_branches = 32;
    static constexpr size_t max_num_moves = 120;
    std::vector<Board>    boards;
    std::vector<DirMoves> dir_moves;
    std::vector<size_t>   num_branches;
    std::vector<std::array<Board,    max_num_branches>> branch_boards;
    std::vector<std::array<DirMoves, max_num_branches>> branch_dir_moves;
    std::vector<std::array<size_t,   max_num_branches>> branch_mobilities;

    SearchStack() : boards(max_num_moves), dir_moves(max_num_moves), num_branches(max_num_moves),
        branch_boards(max_num_moves), branch_dir_moves(max_num_moves), branch_mobilities(max_num_moves)
    {
    }

    SearchStack(const Board& base) : SearchStack()
    {
        init(base);
    }

    void init(const Board& base)
    {
        boards[0]       = base;
        dir_moves[0]    = boards[0].get_moves();
        num_branches[0] = _mm_popcnt_u64(boards[0].move_bits_);
    }
};

inline int end_eval(const Board& board)
{
    auto piece_diff = _mm_popcnt_u64(board.self_bits_) - _mm_popcnt_u64(board.oppo_bits_);
    return piece_diff > 0 ? 1 : piece_diff < 0 ? -1 : 0;
}

inline int end_eval_search_final(SearchStack& stack, size_t index,
    bool last_is_empty, int alpha = int(MIN_SCORE), int beta = int(MAX_SCORE))
{
    auto& this_board = stack.boards[index];
    auto& next_board = stack.boards[index + 1];
    auto possible_moves = ~(this_board.self_bits_ | this_board.oppo_bits_);
    bool any_move = false;
    unsigned int square = 0;
    while (_BitScanForward64(&square, possible_moves))
    {
        possible_moves &= possible_moves - 1u;
        next_board = this_board;
        if (next_board.move_if_valid(Square(square)))
        {
            any_move = true;
            auto score = -end_eval_search_final(stack, index + 1, false, -beta, -alpha);
            if (score >= beta)
                return beta;
            if (score > alpha)
                alpha = score;
        }
    }
    if (any_move)
    {
        return alpha;
    }
    else
    {
        if (last_is_empty)
            return end_eval(this_board);
        else
        {
            next_board = this_board;
            next_board.move_empty();
            auto score = -end_eval_search_final(stack, index + 1, true, -beta, -alpha);
            if (score >= beta)
                return beta;
            if (score > alpha)
                alpha = score;
            return alpha;
        }
    }
}

inline int end_eval_fastest_search(SearchStack& stack, size_t index,
    bool last_is_empty, int alpha = int(MIN_SCORE), int beta = int(MAX_SCORE))
{
    constexpr size_t go_to_final_empties = 2;
    auto& this_board = stack.boards[index];
    auto& next_board = stack.boards[index + 1];
    auto& this_dir_moves = stack.dir_moves[index];
    if (this_board.move_bits_ == 0u)
    {
        if (last_is_empty)
            return end_eval(this_board);
        else
        {
            stack.boards[index + 1] = this_board;
            stack.boards[index + 1].move_empty();
            stack.dir_moves[index + 1] = stack.boards[index + 1].get_moves();
            stack.num_branches[index + 1] = _mm_popcnt_u64(stack.boards[index + 1].move_bits_);
            auto score = -end_eval_fastest_search(stack, index + 1, true, -beta, -alpha);
            if (score >= beta)
                return beta;
            if (score > alpha)
                alpha = score;
            return alpha;
        }
    }
    else if (this_board.empties() <= go_to_final_empties)
    {
        unsigned int square = 0u;
        auto move_bits = this_board.move_bits_;
        auto num_branches = stack.num_branches[index];
        for (size_t i_branch = 0; i_branch < num_branches; ++i_branch)
        {
            _BitScanForward64(&square, move_bits);
            move_bits &= move_bits - 1u;
            next_board = this_board;
            next_board.move(square, this_dir_moves);
            auto score = -end_eval_search_final(stack, index + 1, false, -beta, -alpha);
            if (score >= beta)
                return beta;
            if (score > alpha)
                alpha = score;
        }
        return alpha;
    }
    else
    {
        auto& branch_boards     = stack.branch_boards[index];
        auto& branch_dir_moves  = stack.branch_dir_moves[index];
        auto& branch_mobilities = stack.branch_mobilities[index];
        unsigned int square = 0u;
        auto move_bits = this_board.move_bits_;
        auto num_branches = stack.num_branches[index];
        for (size_t i_branch = 0; i_branch < num_branches; ++i_branch)
        {
            _BitScanForward64(&square, move_bits);
            move_bits &= move_bits - 1u;
            branch_boards[i_branch] = this_board;
            branch_boards[i_branch].move(square, this_dir_moves);
            branch_dir_moves[i_branch] = branch_boards[i_branch].get_moves();
            branch_mobilities[i_branch] = _mm_popcnt_u64(branch_boards[i_branch].move_bits_);
        }

        for (size_t i_branch = 0; i_branch < num_branches; ++i_branch)
        {
            size_t min_mobility = branch_mobilities[i_branch];
            size_t best_branch = i_branch;
            for (size_t j_branch = i_branch + 1u; j_branch < num_branches; ++j_branch)
            {
                if (branch_mobilities[j_branch] < min_mobility)
                {
                    min_mobility = branch_mobilities[j_branch];
                    best_branch = j_branch;
                }
            }
            if (i_branch != best_branch)
            {
                std::swap(branch_boards[i_branch], branch_boards[best_branch]);
                std::swap(branch_dir_moves[i_branch], branch_dir_moves[best_branch]);
                std::swap(branch_mobilities[i_branch], branch_mobilities[best_branch]);
            }
            stack.boards[index + 1] = branch_boards[i_branch];
            stack.dir_moves[index + 1] = branch_dir_moves[i_branch];
            stack.num_branches[index + 1] = branch_mobilities[i_branch];
            auto score = -end_eval_fastest_search(stack, index + 1, false, -beta, -alpha);
            if (score >= beta)
                return beta;
            if (score > alpha)
                alpha = score;
        }
        return alpha;
    }
}

inline float end_eval_with_empties(const Board& board, Square prev_move)
{
    thread_local SearchStack stack;
    stack.init(board);
    auto eval = end_eval_fastest_search(stack, 0, prev_move == EMPTY_MOVE);
    return float(eval);
}

template<size_t MaxParallel, bool Neural = true>
struct Evaluator
{
    static constexpr size_t max_parallel  = MaxParallel;
    nn::Net net;
    std::vector<float> input_images;
    std::vector<float> output_values;
    std::vector<float> output_policies;

    Evaluator(std::string netfile) :
        net(netfile),
        input_images(max_parallel * 64),
        output_values(max_parallel),
        output_policies(max_parallel * 64)
    {
        if (global_options.target_device == TargetDevice::CUDA)
            net.to_gpu();
    }

    template<typename Move>
    void evaluate_neural(Board** boards, const size_t n_boards, float** values, Move** moves, float** priors)
    {
        for (size_t i = 0; i < n_boards; ++i)
            for (size_t s = 0; s < 64; ++s)
            {
                const auto mask = uint64_t(1) << s;
                input_images[i * 64 + s] = float(int(bool(boards[i]->self_bits_ & mask)) -
                    int(bool(boards[i]->oppo_bits_ & mask)));
            }
        
        if (!global_options.play_random)
        {
            net.predict(input_images.data(), n_boards,
                output_values.data(), output_policies.data());
        }
        else
        {
            std::fill(output_values.begin(), output_values.end(), 0.0f);
            std::fill(output_policies.begin(), output_policies.end(), 0.0f);
        }
        
        for (size_t i = 0; i < n_boards; ++i)
        {
            *values[i] = output_values[i];
            size_t n_moves = boards[i]->num_moves();
            float max_prior = 0.0;
            auto* __restrict priors_i = priors[i];
            auto* __restrict moves_i  = moves[i];
            if (!priors_i || !moves_i)
                throw std::logic_error("");
            for (size_t j = 0; j < n_moves; ++j)
            {
                priors_i[j] = output_policies[i * 64 + moves_i[j]];
                max_prior = std::max(max_prior, priors_i[j]);
            }
            float total_prior = 0.0;
            for (size_t j = 0; j < n_moves; ++j)
            {
                priors_i[j] = std::exp((priors_i[j] - max_prior) / global_options.policy_temp);
                total_prior += priors_i[j];
            }
            float inv_total_prior = 1.0f / total_prior;
            for (size_t j = 0; j < n_moves; ++j)
            {
                priors_i[j] *= inv_total_prior;
            }
        }
    }

    template<typename Move>
    void evaluate_mobility(Board** boards, const size_t n_boards, float** values, Move** moves, float** priors)
    {
        for (size_t i = 0; i < n_boards; ++i)
        {
            Board self(boards[i]->self_bits_, boards[i]->oppo_bits_);
            Board oppo(boards[i]->oppo_bits_, boards[i]->self_bits_);
            self.get_moves();
            oppo.get_moves();
            *values[i] = std::tanh(0.2f * float(int64_t(self.num_moves() - oppo.num_moves())));
            const auto n_moves = boards[i]->num_moves();
            for (size_t j = 0; j < n_moves; ++j)
                priors[i][j] = 1.0f / float(n_moves);
        }
    }

    template<typename Move>
    auto evaluate(Board** boards, const size_t n_boards, float** values, Move** moves, float** priors)
    {
        if (n_boards > max_parallel)
            throw std::logic_error("");
        if constexpr (Neural)
            evaluate_neural(boards, n_boards, values, moves, priors);
        else
            evaluate_mobility(boards, n_boards, values, moves, priors);
    }
};

struct TreeNode
{
    int32_t visits = 0u;
    Square  prev_move;
    bool    is_expanded = false;
    bool    is_exact = false;
    bool    will_be_exact = false;
    float   value_sum = 0.f;
    float   prior;
    std::vector<TreeNode> children{};

    TreeNode(Square prev_move, float prior) :
        prev_move{prev_move}, prior{prior}
    {
    }

    void add_value(float value)
    {
        value_sum += value;
        visits += 1;
    }

    void add_virtual_loss()
    {
        if (!is_exact)
        {
            visits += 1;
            value_sum -= 1.0f;
        }
    }

    void remove_virtual_loss()
    {
        visits -= 1;
        value_sum += 1.0f;
    }

    float value_if_exact() const
    {
        if (!is_exact)
            throw std::logic_error("");
        return value_sum > 0 ? 1.0f : value_sum < 0 ? -1.0f : 0.0f;
    }

    auto get_visits() const
    {
        std::vector<std::pair<Square, float>> visits;
        visits.reserve(children.size());
        for (const auto& child : children)
            visits.push_back(std::make_pair(child.prev_move, float(child.visits)));
        return visits;
    }

    int eval_cp(bool black_to_move) const
    {
        double winrate = (double(value_sum) / double(visits) + 1.0) / 2.0;
        if (!black_to_move)
            winrate = 1.0 - winrate;
        constexpr double winrate_threshold = 0.00317045;
        winrate = std::max(winrate_threshold, winrate);
        winrate = std::min(1.0 - winrate_threshold, winrate);
        return int(std::round(-400.0 * std::log10(1.0 / winrate - 1.0)));
    }

    std::string eval_cp_string(bool black_to_move) const
    {
        auto cp = eval_cp(black_to_move);
        char str[] = "(+0.00)";
        str[1] = cp > 0 ? '+' : cp < 0 ? '-' : ' ';
        str[2] += (std::abs(cp) / 100);
        str[4] += (std::abs(cp) % 100) / 10;
        str[5] += std::abs(cp) % 10;
        return std::string(str);
    }

    void print(bool black_to_move, size_t min_visits = 1000) const
    {
        if (is_exact)
        {
            std::cout << "  exact value: " << eval_cp_string(black_to_move) << "\n";
            return;
        }
        else if (!is_expanded)
        {
            std::cout << "  leaf node reached." << "\n";
            return;
        }
        std::vector<std::pair<size_t, double>> children_visits;
        double total_visits = 0;
        for (size_t i = 0; i < children.size(); ++i)
        {
            children_visits.push_back(std::make_pair(i, double(children[i].visits)));
            total_visits += double(children[i].visits);
        }
        if (total_visits < min_visits)
        {
            return;
        }
        total_visits = std::max(1.0, total_visits);
        std::sort(children_visits.begin(), children_visits.end(),
            [](auto& x, auto& y) { return x.second > y.second; });
        for (size_t i = 0; i < children.size(); ++i)
            children_visits[i].second /= total_visits;
        std::cout << std::setw(10) << size_t(total_visits) << "   " << eval_cp_string(black_to_move) << "   ";
        for (size_t i = 0; i < 4 && i < children.size(); ++i)
            std::cout << square_name(children[children_visits[i].first].prev_move) <<
            "(" << std::setw(2) << int(children_visits[i].second * 100) << ")   ";
        std::cout << "\n";
        children[children_visits[0].first].print(!black_to_move, min_visits);
    }
};

inline float ucb_parent_factor(uint32_t total_visits)
{
    float count = float(total_visits);
    float pb_c = log1pf(count / global_options.cpuct_base) + global_options.cpuct;
    return pb_c * sqrtf(count);
}

struct TreeWorker
{
    struct board_value_pending_tag {};
    struct board_value_is_exact_tag {};
    using board_value_t = std::variant<float, std::future<float>,
        board_value_pending_tag, board_value_is_exact_tag>;

    std::vector<Board>                  boards_array;
    std::vector<std::array<Square, 32>> moves_array;
    std::vector<std::array<float, 32>>  priors_array;
    std::vector<board_value_t>          values_array;
    std::vector<std::vector<TreeNode*>> nodes_array;

    TreeWorker(const TreeWorker&) = delete;

    void pre_uct(Board board, TreeNode& root)
    {
        std::vector<TreeNode*> nodes;
        nodes.reserve(60);
        auto* current_node = &root;
        board_value_t board_value{0.0f};
        for (;;)
        {
            if (current_node->is_exact)
            {
                board_value = board_value_is_exact_tag{};
                auto leaf_value = current_node->value_if_exact();
                current_node->add_value(leaf_value);
                break;
            }
            else if (current_node->will_be_exact)
            {
                board_value = board_value_pending_tag{};
                break;
            }
            else if (board.empties() <= global_options.exact_empties)
            {
                if (board.empties() == 0u) // game has ended
                {
                    current_node->is_exact = true;
                    board_value = board_value_is_exact_tag{};
                    auto leaf_value = float(end_eval(board));
                    current_node->add_value(leaf_value);
                }
                else
                {
                    current_node->will_be_exact = true;
                    board_value = std::async(std::launch::async,
                        end_eval_with_empties, board, current_node->prev_move);
                }
                break;
            }
            else if (current_node->is_expanded)
            {
                current_node->add_virtual_loss();
                nodes.push_back(current_node);
                const auto num_children = current_node->children.size();
                auto& children = current_node->children;
                if (children.size() == 0u)
                {
                    board.print();
                    throw std::logic_error("");
                }
                TreeNode* best_child = &children[0];
                if (children.size() >= 2u)
                {
                    const float cpuct = ucb_parent_factor(current_node->visits);
                    float best_score = std::numeric_limits<float>::lowest();
                    for (size_t i = 0; i < num_children; ++i)
                    {
                        auto& child = children[i];
                        auto fpu = (current_node == &root) ? 1.0f : -1.0f;
                        auto N = float(child.visits);
                        auto P = float(child.prior);
                        auto QN = float(child.value_sum);
                        auto score = (N == 0) ? cpuct * P + fpu : (cpuct / (1.0f + N)) * P - (QN / N);
                        if (score > best_score)
                        {
                            best_score = score;
                            best_child = &child;
                        }
                    }
                }
                board.move(best_child->prev_move);
                current_node = best_child;
            }
            else
            {
                board.get_moves();
                auto num_moves = board.num_moves();
                if (num_moves >= 2u)
                    break;
                else if (num_moves == 1u)
                {
                    Square move = EMPTY_MOVE;
                    board.get_moves_list(&move);
                    current_node->is_expanded = true;
                    current_node->children.push_back(TreeNode(move, 1.0f));
                    current_node->add_virtual_loss();
                    nodes.push_back(current_node);
                    current_node = &(current_node->children.front());
                    board.move_if_valid(move);
                }
                else if (current_node->prev_move != EMPTY_MOVE)
                {
                    Square move = EMPTY_MOVE;
                    board.get_moves_list(&move);
                    current_node->is_expanded = true;
                    current_node->children.push_back(TreeNode(move, 1.0f));
                    current_node->add_virtual_loss();
                    nodes.push_back(current_node);
                    current_node = &(current_node->children.front());
                    board.move_empty();
                }
                else
                {
                    current_node->is_exact = true;
                    auto leaf_value = float(end_eval(board));
                    current_node->add_value(leaf_value);
                    break;
                }
            }
        }
        current_node->add_virtual_loss();
        nodes.push_back(current_node);
        std::reverse(nodes.begin(), nodes.end());
        std::array<Square, 32> moves;
        board.get_moves_list(moves.data());
        boards_array.push_back(board);
        values_array.emplace_back(std::move(board_value));
        moves_array.push_back(moves);
        nodes_array.push_back(std::move(nodes));
    }

    template<typename Any>
    void call_eval(Any& evaluator)
    {
        auto n_boards = boards_array.size();
        priors_array.resize(n_boards);
        values_array.resize(n_boards);
        if (n_boards > Any::max_parallel)
            throw std::logic_error("");
        std::array<Board*,  Any::max_parallel> boards_ptrs{};
        std::array<float*,  Any::max_parallel> values_ptrs{};
        std::array<Square*, Any::max_parallel> moves_ptrs{};
        std::array<float*,  Any::max_parallel> priors_ptrs{};
        size_t i = 0u;
        for (size_t j = 0; j < n_boards; ++j)
        {
            if (nodes_array[j].front()->is_exact || nodes_array[j].front()->will_be_exact)
                continue;
            boards_ptrs[i] = &boards_array[j];
            values_ptrs[i] = &std::get<float>(values_array[j]);
            moves_ptrs[i]  = moves_array[j].data();
            priors_ptrs[i] = priors_array[j].data();
            ++i;
        }
        if (i > 0)
        {
            evaluator.evaluate(
                boards_ptrs.data(), i, values_ptrs.data(), moves_ptrs.data(), priors_ptrs.data());
        }
    }

    void post_uct()
    {
        for (size_t i = 0; i < boards_array.size(); ++i)
        {
            auto& board      = boards_array[i];
            auto  num_moves  = board.num_moves();
            auto& moves      = moves_array[i];
            auto& priors     = priors_array[i];
            auto& leaf_value = values_array[i];
            auto& nodes      = nodes_array[i];

            auto  node_iter = nodes.begin();
            if ((*node_iter)->is_exact)
            {
                if (std::holds_alternative<board_value_is_exact_tag>(leaf_value))
                {
                    leaf_value = (*node_iter)->value_if_exact();
                    leaf_value = -std::get<float>(leaf_value);
                    ++node_iter;
                }
                else if (std::holds_alternative<board_value_pending_tag>(leaf_value))
                {
                    leaf_value = (*node_iter)->value_if_exact();
                    // virtual loss is removed already
                    (*node_iter)->add_value(std::get<float>(leaf_value));
                    leaf_value = -std::get<float>(leaf_value);
                    ++node_iter;
                }
            }
            else if ((*node_iter)->will_be_exact)
            {
                leaf_value = std::get<std::future<float>>(leaf_value).get();
                (*node_iter)->will_be_exact = false;
                (*node_iter)->is_exact = true;
                (*node_iter)->visits = 0;
                (*node_iter)->value_sum = 0.0f;
                (*node_iter)->add_value(std::get<float>(leaf_value));
                leaf_value = -std::get<float>(leaf_value);
                ++node_iter;
            }
            else if (!(*node_iter)->is_expanded)
            {
                (*node_iter)->is_expanded = true;
                (*node_iter)->children.reserve(num_moves);
                for (size_t j = 0; j < num_moves; ++j)
                    (*node_iter)->children.push_back(TreeNode(moves[j], priors[j]));
            }
            for (; node_iter != nodes.end(); ++node_iter, leaf_value = -std::get<float>(leaf_value))
            {
                (*node_iter)->remove_virtual_loss();
                (*node_iter)->add_value(std::get<float>(leaf_value));
            }
        }
        boards_array.resize(0);
        values_array.resize(0);
        moves_array.resize(0);
        nodes_array.resize(0);
    }
};

inline void regularize_visits(std::vector<std::pair<Square, float>>& visits)
{
    float max_visits = 0.0f;
    for (const auto& visit : visits)
        max_visits = std::max(max_visits, visit.second);
    for (auto& visit : visits)
        visit.second /= max_visits;
}

inline auto pick_move_from_visits(
    const std::vector<std::pair<Square, float>>& in_visits, size_t empties,
    std::optional<float> maybe_temp = std::nullopt)
{
    auto visits = in_visits;
    if (visits.size() == 0u)
        throw std::logic_error("");
    if (visits.size() == 1u)
        return visits[0].first;

    /*auto temperature = maybe_temp ? maybe_temp.value() : 
        empties > global_options.midgame_empties ? global_options.opening_temp :
        empties > global_options.endgame_empties ? global_options.midgame_temp :
            global_options.endgame_temp;*/

    auto temperature = std::max(global_options.temp_final,
        global_options.temp_init - global_options.temp_slope * float(60 - int(empties)));

    float total_visits = 0.0f;
    if (temperature > 0.0001f)
    {
        for (auto& visit : visits)
        {
            total_visits += std::pow(visit.second, 1.0f / temperature);
            visit.second = total_visits;
        }
    }
    else
    {
        for (auto& visit : visits)
        {
            total_visits += visit.second > 0.9999f ? visit.second : 0.0f;
            visit.second = total_visits;
        }
    }
    
    Square picked_move = INVALID_MOVE;
    auto rand = std::uniform_real_distribution<float>(0., total_visits)(engine);
    for (const auto& visit : visits)
    {
        if (rand <= visit.second)
        {
            picked_move = visit.first;
            break;
        }
    }
    return picked_move;
}

inline std::vector<float> dirichlet_noise(size_t n)
{
    auto dist = std::gamma_distribution<float>(global_options.dirichlet_alpha, 1.0);
    std::vector<float> noises(n);
    float total_noise = 0.0f;
    for (auto& noise : noises)
    {
        noise = dist(engine);
        total_noise += noise;
    }
    for (auto& noise : noises)
        noise /= total_noise;
    return noises;
}

struct Searcher
{
    Evaluator<global_max_parallel, true>* evaluator_ptr_ = nullptr;
    Board board_;
    TreeWorker tree_worker_;
    TreeNode root_;
    bool board_is_ready_ = false;
    bool result_is_ready_ = false;
    
    Searcher(Evaluator<global_max_parallel, true>& evaluator) :
        evaluator_ptr_{&evaluator}, tree_worker_{}, root_(INVALID_MOVE, 1.0f)
    {
    }

    void set_board(const Board& board, Square prev_move)
    {
        board_ = board;
        root_ = TreeNode(prev_move, 1.0f); 
        board_is_ready_ = true;
        result_is_ready_ = false;
    }

    void move(Square prev_move)
    {
        board_.move(prev_move);
        auto new_root = TreeNode(prev_move, 1.0f);
        if (!global_options.independent_tree && root_.is_expanded)
        {
            for (auto& child : root_.children)
                if (child.prev_move == prev_move)
                    new_root = std::move(child);
        }
        root_ = new_root;
        board_is_ready_ = true;
        result_is_ready_ = false;
    }

    template<bool AddDirichletNoise = false>
    void go(size_t num_nodes)
    {
        if (!board_is_ready_)
            throw std::logic_error("");

        auto& children = root_.children;
        std::vector<float> kld_q;
        std::vector<float> kld_p;
        size_t kld_last_num_nodes = num_nodes;

        if (root_.is_expanded)
        {
            kld_q.resize(children.size());
            kld_p.resize(children.size());

            float total_visits = 0;
            for (size_t i = 0; i < children.size(); ++i)
                total_visits += (kld_p[i] = float(children[i].visits));
            for (size_t i = 0; i < children.size(); ++i)
            {
                kld_p[i] /= total_visits;
            }
        }
        else
        {
            tree_worker_.pre_uct(board_, root_);
            tree_worker_.call_eval(*evaluator_ptr_);
            tree_worker_.post_uct();
            
            kld_q.resize(children.size());
            kld_p = std::vector<float>(children.size(), 1.0f / children.size());

            if constexpr (AddDirichletNoise)
            {
                auto noises = dirichlet_noise(children.size());
                for (size_t i = 0; i < children.size(); ++i)
                    children[i].prior = (1.0f - global_options.dirichlet_weight) * children[i].prior +
                        global_options.dirichlet_weight * noises[i];
            }
        }
        size_t initial_visits = root_.visits;
        
        while (num_nodes > 0u)
        {
            for (size_t j = 0; j < global_max_parallel && num_nodes > 0u; ++j, --num_nodes)
                tree_worker_.pre_uct(board_, root_);
            tree_worker_.call_eval(*evaluator_ptr_);
            tree_worker_.post_uct();
            if (global_options.kldgain_limit > 0.0f &&
                num_nodes + global_options.kldgain_interval <= kld_last_num_nodes)
            {
                kld_last_num_nodes = num_nodes;
                std::swap(kld_q, kld_p);
                float total_visits = 0;
                float kld = 0.0f;
                for (size_t i = 0; i < children.size(); ++i)
                    total_visits += (kld_p[i] = float(children[i].visits));
                for (size_t i = 0; i < children.size(); ++i)
                {
                    kld_p[i] /= total_visits;
                    kld += kld_p[i] * std::log(kld_p[i] / kld_q[i]);
                }
                if (kld < global_options.kldgain_limit)
                {
                    //printf("nodes = %d\tvisits = %d\n", int(root_.visits - initial_visits), int(root_.visits));
                    break;
                }
            }
        }

        board_is_ready_ = false;
        result_is_ready_ = true;
    }

    auto get_results() const
    {
        if (!result_is_ready_)
            throw std::logic_error("");
        auto visits = root_.get_visits();
        regularize_visits(visits);
        return std::make_pair(visits, pick_move_from_visits(visits, board_.empties()));
    }

    const auto& root() const
    {
        if (!result_is_ready_)
            throw std::logic_error("");
        return root_;
    }

};
