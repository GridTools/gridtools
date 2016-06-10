#pragma once
#include "stencil_composition/timer.hpp"

namespace gridtools {

    /**
    * @class timer_cuda
    * CUDA implementation of the Timer interface
    */
    class timer_cuda : public timer< timer_cuda > // CRTP
    {
      public:
        __host__ timer_cuda(std::string name) : timer< timer_cuda >(name) {
            // create the CUDA events
            cudaEventCreate(&start_);
            cudaEventCreate(&stop_);
        }
        __host__ ~timer_cuda() {
            // free the CUDA events
            cudaEventDestroy(start_);
            cudaEventDestroy(stop_);
        }

        /**
        * Reset counters
        */
        __host__ void reset_impl() {}

        /**
        * Start the stop watch
        */
        __host__ void start_impl() {
            // insert a start event
            cudaEventRecord(start_, 0);
        }

        /**
        * Pause the stop watch
        */
        __host__ double pause_impl() {
            // insert stop event and wait for it
            cudaEventRecord(stop_, 0);
            cudaEventSynchronize(stop_);

            // compute the timing
            float result;
            cudaEventElapsedTime(&result, start_, stop_);
            return static_cast< double >(result) * 0.001f; // convert ms to s
        }

      private:
        cudaEvent_t start_;
        cudaEvent_t stop_;
    };
}
