#pragma once
#include <cstdint>
#include <string>

enum class Mode
{
    Selfplay, Compare, Interactive, Invalid
};

enum class TargetDevice
{
    CUDA, CPU
};

struct Options
{
    Mode         mode             = Mode::Invalid;
    TargetDevice target_device    = TargetDevice::CUDA;
    bool         play_random      = false;
    size_t       games            = 1;
    size_t       playouts         = 200;
    std::string  net_path         = "";
    std::string  alt_net_path     = "";
    std::string  games_path       = "";
    bool         print_moves      = false;
    float        cpuct            = 2.4f;
    float        cpuct_base       = 19652.0f;
    float        dirichlet_alpha  = 1.0f;
    float        dirichlet_weight = 0.5f;
    size_t       exact_empties    = 6;
    float        temp_init        = 1.0f;
    float        temp_slope       = 0.05f;
    float        temp_final       = 0.0f;
    float        policy_temp      = 1.0f;
    float        kldgain_limit    = 0.0f;
    size_t       kldgain_interval = 100;
    bool         independent_tree = false;
};

constexpr size_t global_max_parallel = 20u;

extern Options global_options;

inline auto parse_options(char** argv, char** argv_end)
{
    std::string option;

    auto get_next = [&]()
    {
        if (argv >= argv_end)
            throw std::logic_error("error reading option " + option);
        return std::string(*argv++);
    };

    auto ranged = [&](auto&& val, const auto& min, const auto& max)
    {
        if (min <= val && val <= max)
            return std::forward<decltype(val)>(val);
        else
            throw std::logic_error(option + " is out of range");
    };

    get_next(); // skip program name
    while (argv < argv_end)
    {
#define OPTION_CASE(name, member, value) \
        else if (option == name) global_options.member = value;
        
        option = get_next();
        if (option == "--mode")
        {
            auto value = get_next();
            if      (value == "selfplay")    global_options.mode = Mode::Selfplay;
            else if (value == "compare")     global_options.mode = Mode::Compare;
            else if (value == "interactive") global_options.mode = Mode::Interactive;
        }
        else if (option == "--device")
        {
            if (get_next() == "cpu")   global_options.target_device = TargetDevice::CPU;
        }
        OPTION_CASE("--play-random",      play_random,      true)
        OPTION_CASE("--games",            games,            ranged(std::stoll(get_next()), 1, 1e9))
        OPTION_CASE("--playouts",         playouts,         ranged(std::stoll(get_next()), (long long)global_max_parallel, 1e9))
        OPTION_CASE("--net-path",         net_path,         get_next())
        OPTION_CASE("--alt-net-path",     alt_net_path,     get_next())
        OPTION_CASE("--games-path",       games_path,       get_next())
        OPTION_CASE("--print-moves",      print_moves,      true)
        OPTION_CASE("--cpuct",            cpuct,            ranged(std::stof(get_next()), 0.0f, 1000.0f))
        OPTION_CASE("--cpuct-base",       cpuct_base,       ranged(std::stof(get_next()), 1.0f, 1e9f))
        OPTION_CASE("--dirichlet-alpha",  dirichlet_alpha,  ranged(std::stof(get_next()), 0.001f, 1000.0f))
        OPTION_CASE("--dirichlet-weight", dirichlet_weight, ranged(std::stof(get_next()), 0.001f, 1000.0f))
        OPTION_CASE("--exact-empties",    exact_empties,    ranged(std::stoll(get_next()), 0, 24))
        OPTION_CASE("--temp-init",        temp_init,        ranged(std::stof(get_next()), 0.0f, 1000.0f))
        OPTION_CASE("--temp-slope",       temp_slope,       ranged(std::stof(get_next()), 0.0f, 1000.0f))
        OPTION_CASE("--temp-final",       temp_final,       ranged(std::stof(get_next()), 0.0f, 1000.0f))
        OPTION_CASE("--policy-temp",      policy_temp,      ranged(std::stof(get_next()), 0.001f, 100.0f))
        OPTION_CASE("--kldgain-limit",    kldgain_limit,    ranged(std::stof(get_next()), 0.0f, 1.0f))
        OPTION_CASE("--kldgain-interval", kldgain_interval, ranged(std::stof(get_next()), 1, 10000))
        OPTION_CASE("--independent-tree", independent_tree, true)
        else std::cerr << ("ignore unknown option " + option + "\n");
    }
}
