#pragma once

#include <sstream>
#include <string>

namespace gridtools {

    /**
    * @class Timer
    * Measures total elapsed time between all start and stop calls
    */
    template < typename TimerImpl >
    class timer {
        DISALLOW_COPY_AND_ASSIGN(timer);

      protected:
        __host__ timer(std::string name) {
            m_name = name;
            reset();
        }
        __host__ ~timer() {}

      public:
        /**
        * Reset counters
        */
        __host__ void reset() {
            m_total_time = 0.0;
            static_cast< TimerImpl * >(this)->reset_impl();
        }

        /**
        * Start the stop watch
        */
        __host__ void start() { static_cast< TimerImpl * >(this)->start_impl(); }

        /**
        * Pause the stop watch
        */
        __host__ void pause() { m_total_time += static_cast< TimerImpl * >(this)->pause_impl(); }

        /**
        * @return total elapsed time [s]
        */
        __host__ double total_time() const { return m_total_time; }

        /**
        * @return total elapsed time [s] as string
        */
        __host__ std::string to_string() const {
            std::ostringstream out;
            if (m_total_time < 0)
                out << "\t[s]\t" << m_name << "NO_TIMES_AVAILABLE";
            else
                out << m_name << "\t[s]\t" << m_total_time;
            return out.str();
        }

      private:
        std::string m_name;
        double m_total_time;
    };
}
