/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../host_device.hpp"
#include <cmath>
#include <sstream>
#include <string>
#include <utility>

namespace gridtools {

    /**
     * @class Timer
     * Measures total elapsed time between all start and stop calls
     */
    template <typename TimerImpl>
    class timer {
      protected:
        timer(std::string name) : m_name(std::move(name)) {}

      public:
        /**
         * Reset counters
         */
        GT_FUNCTION_HOST void reset() {
            m_total_time = 0;
            m_counter = 0;
        }

        /**
         * Start the stop watch
         */
        GT_FUNCTION_HOST void start() { impl().start_impl(); }

        /**
         * Pause the stop watch
         */
        GT_FUNCTION_HOST void pause() {
            m_total_time += impl().pause_impl();
            m_counter++;
        }

        /**
         * @return total elapsed time [s]
         */
        GT_FUNCTION_HOST double total_time() const { return m_total_time; }

        /**
         * @return how often the timer was paused
         */
        GT_FUNCTION_HOST size_t count() const { return m_counter; }

        /**
         * @return total elapsed time [s] as string
         */
        GT_FUNCTION_HOST std::string to_string() const {
            std::ostringstream out;
            if (m_total_time < 0 || std::isnan(m_total_time))
                out << m_name << "\t[s]\t"
                    << "NO_TIMES_AVAILABLE"
                    << " (" << m_counter << "x called)";
            else
                out << m_name << "\t[s]\t" << m_total_time << " (" << m_counter << "x called)";
            return out.str();
        }

      private:
        TimerImpl &impl() { return *static_cast<TimerImpl *>(this); }

        std::string m_name;
        double m_total_time = 0;
        size_t m_counter = 0;
    };
} // namespace gridtools
