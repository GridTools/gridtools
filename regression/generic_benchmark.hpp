/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <gridtools/common/timer/timer_traits.hpp>

#include <functional>
#include <string>

namespace {
    template <typename Backend>
    class generic_benchmark {
      public:
        template <typename F>
        generic_benchmark(F &&f) : m_f(f), m_meter("") {}

        void run() {
            m_meter.start();
            m_f();
            m_meter.pause();
        }
        void reset_meter() { m_meter.reset(); }
        std::string print_meter() { return m_meter.to_string(); }

      private:
        std::function<void()> m_f;

        using performance_meter_t = typename gridtools::timer_traits<Backend>::timer_type;
        performance_meter_t m_meter;
    };
} // namespace
