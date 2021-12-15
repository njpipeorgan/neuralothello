
// requires openmp
// not thread safe

#pragma once

#include <cstdio>
#include <tuple>
#include <vector>
#include <omp.h>

#define deploy_timer() \
	static std::vector<double> _t_timer_ticks; \
	static std::tuple<double, const char*> _t_timer_config; \
	std::get<0>(_t_timer_config) = 0.001; \
	std::get<1>(_t_timer_config) = "ms";

#define configure_timer(multiplier, unit_name_str) \
	std::get<0>(_t_timer_config) = multiplier; \
	std::get<1>(_t_timer_config) = unit_name_str; \

#define tick() \
	_t_timer_ticks.push_back(omp_get_wtime());

#define tock() \
	{ \
		auto t2 = omp_get_wtime(); \
		auto t1 = _t_timer_ticks.back(); \
		printf("time: %.3f %s\n", (t2 - t1) / std::get<0>(_t_timer_config), std::get<1>(_t_timer_config)); \
		_t_timer_ticks.pop_back(); \
	}
