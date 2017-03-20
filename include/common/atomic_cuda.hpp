/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once
#include "cuda_runtime.h"

namespace gridtools {
    /**
    * @class atomic_cuda
    * generic implementation for CUDA that provides atomic functions
    */
    template < typename T >
    class atomic_cuda {
      public:
        /**
        * Function computing an atomic addition
        * @param var reference to variable where the addition is performed
        * @param val value added to var
        * @return the old value contained in var
        */
        __device__ static T atomic_add(T &var, const T val) { return ::atomicAdd(&var, val); }

        /**
        * Function computing an atomic substraction
        * @param var reference to variable where the substracion is performed
        * @param val value added to var
        * @return the old value contained in var
        */
        __device__ static T atomic_sub(T &var, const T val) { return ::atomicSub(&var, val); }

        /**
        * Function computing an atomic exchange of value of a variable
        * @param var reference to variable which value is replaced by val
        * @param val value inserted in variable var
        * @return the old value contained in var
        */
        __device__ static T atomic_exch(T &var, const T val) { return ::atomicExch(&var, val); }

        /**
        * Function computing an atomic min operation
        * @param var reference used to compute and store the min
        * @param val value used in the min comparison
        * @return the old value contained in var
        */
        __device__ static T atomic_min(T &var, const T val) { return ::atomicMin(&var, val); }

        /**
        * Function computing an atomic max operation
        * @param var reference used to compute and store the min
        * @param val value used in the min comparison
        * @return the old value contained in var
        */
        __device__ static T atomic_max(T &var, const T val) { return ::atomicMax(&var, val); }
    };

    /**
    * @class atomic_cuda - Specialization for float
    * generic implementation for CUDA that provides atomic functions
    */
    template <>
    class atomic_cuda< float > {
      public:
        /**
        * Function computing an atomic addition
        * @param var reference to variable where the addition is performed
        * @param val value added to var
        * @return the old value contained in var
        */
        __device__ static float atomic_add(float &var, const float val) { return ::atomicAdd(&var, val); }

        /**
        * Function computing an atomic substraction
        * @param var reference to variable where the substracion is performed
        * @param val value added to var
        * @return the old value contained in var
        */
        __device__ static float atomic_sub(float &var, const float val) { return ::atomicAdd(&var, -val); }

        /**
        * Function computing an atomic exchange of value of a variable
        * @param var reference to variable which value is replaced by val
        * @param val value inserted in variable var
        * @return the old value contained in var
        */
        __device__ static float atomic_exch(float &var, const float val) { return ::atomicExch(&var, val); }

        /**
        * Function computing an atomic min operation
        * @param var reference used to compute and store the min
        * @param val value used in the min comparison
        * @return the old value contained in var
        */
        __device__ static float atomic_min(float &var, const float val) {
            float old = var;
            float assumed;
            if (old <= val)
                return old;
            do {
                assumed = old;
                old = __int_as_float(atomicCAS((unsigned int *)(&var), __float_as_int(assumed), __float_as_int(val)));
            } while (old != assumed && old > val);

            return old;
        }

        /**
        * Function computing an atomic max operation
        * @param var reference used to compute and store the min
        * @param val value used in the min comparison
        * @return the old value contained in var
        */
        __device__ static float atomic_max(float &var, const float val) {
            float old = var;
            float assumed;
            if (old >= val)
                return old;
            do {
                assumed = old;
                old = __int_as_float(atomicCAS((unsigned int *)(&var), __float_as_int(assumed), __float_as_int(val)));
            } while (old != assumed && old < val);

            return old;
        }
    };

    /**
    * @class AtomicCUDA  - specialization for double
    * specialization for doubles of AtomicCUDA that provides atomic functions
    */
    template <>
    class atomic_cuda< double > {
      public:
        /**
        * Function computing an atomic addition
        * @param var reference to variable where the addition is performed
        * @param val value added to var
        * @return the old value contained in var
        */
        __device__ static double atomic_add(double &var, const double val) {
            unsigned long long int *address_as_ull = (unsigned long long int *)(&var);
            unsigned long long int old = *address_as_ull, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
            } while (assumed != old);
            return __longlong_as_double(old);
        }

        /**
        * Function computing an atomic substraction
        * @param var reference to variable where the substracion is performed
        * @param val value added to var
        * @return the old value contained in var
        */
        __device__ static double atomic_sub(double &var, const double val) {
            unsigned long long int *address_as_ull = (unsigned long long int *)(&var);
            unsigned long long int old = *address_as_ull, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed, __double_as_longlong(__longlong_as_double(assumed) - val));
            } while (assumed != old);
            return __longlong_as_double(old);
        }

        /**
        * Function computing an atomic exchange of value of a variable
        * @param var reference to variable which value is replaced by val
        * @param val value inserted in variable var
        * @return the old value contained in var
        */
        __device__ static double atomic_exch(double &x, const double val) {
            unsigned long long int *address_as_ull = (unsigned long long int *)(&x);
            unsigned long long int old = *address_as_ull, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
            } while (assumed != old);
            return __longlong_as_double(old);
        }

        /**
        * Function computing an atomic min operation
        * @param var reference used to compute and store the min
        * @param val value used in the min comparison
        * @return the old value contained in var
        */
        __device__ static double atomic_min(double &var, const double val) {
            unsigned long long int *address_as_ull = (unsigned long long int *)(&var);

            double old = var;
            double assumed;
            if (old <= val)
                return old;
            do {
                assumed = old;
                old = __longlong_as_double(
                    atomicCAS(address_as_ull, __double_as_longlong(assumed), __double_as_longlong(val)));
            } while (old != assumed && old > val);

            return old;
        }

        /**
        * Function computing an atomic min operation
        * @param var reference used to compute and store the min
        * @param val value used in the min comparison
        * @return the old value contained in var
        */
        __device__ static double atomic_max(double &var, const double val) {
            unsigned long long int *address_as_ull = (unsigned long long int *)(&var);

            double old = var;
            double assumed;
            if (old >= val)
                return old;
            do {
                assumed = old;
                old = __longlong_as_double(
                    atomicCAS(address_as_ull, __double_as_longlong(assumed), __double_as_longlong(val)));
            } while (old != assumed && old < val);

            return old;
        }
    };

    template < typename T >
    struct get_atomic_helper {
        typedef atomic_cuda< T > type;
    };
} // namespace gridtools
