/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../defs.hpp"
#include "timer.hpp"
#include <limits>
#include <string>

namespace gridtools {

    /**
     * @class timer_omp
     */
    class timer_omp : public timer<timer_omp> // CRTP
    {
      public:
        timer_omp(std::string name) : timer<timer_omp>(name) { startTime_ = 0.0; }
        ~timer_omp() {}

        void set_impl(double const &time_) { startTime_ = time_; }

        void start_impl() {
#if defined(_OPENMP)
            startTime_ = omp_get_wtime();
#endif
        }

        double pause_impl() {
#if defined(_OPENMP)
            return omp_get_wtime() - startTime_;
#else
            return std::numeric_limits<double>::quiet_NaN();
#endif
        }

      private:
        double startTime_;
    };
} // namespace gridtools
