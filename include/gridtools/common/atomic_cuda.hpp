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

#include "cuda_runtime.hpp"
#include "host_device.hpp"

#ifdef __HIPCC__
GT_FUNCTION_HOST float __int_as_float(int) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST int __float_as_int(float) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST double __longlong_as_double(long long) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST long long __double_as_longlong(double) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST int atomicAdd(int *, int) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST int atomicSub(int *, int) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST int atomicExch(int *, int) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST int atomicMin(int *, int) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST int atomicMax(int *, int) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST float atomicAdd(float *, float) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST float atomicSub(float *, float) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST float atomicExch(float *, float) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST float atomicMin(float *, float) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST float atomicMax(float *, float) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST double atomicAdd(double *, double) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST double atomicSub(double *, double) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST double atomicExch(double *, double) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST double atomicMin(double *, double) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST double atomicMax(double *, double) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST unsigned atomicCAS(unsigned *, unsigned, unsigned) {
    assert(false);
    return 0;
}
GT_FUNCTION_HOST unsigned long long atomicCAS(unsigned long long *, unsigned long long, unsigned long long) {
    assert(false);
    return 0;
}
#endif

namespace gridtools {
    /** \ingroup common
        @{
        \ingroup atomic
        @{
    */

    /**
     * @class atomic_cuda
     * generic implementation for CUDA that provides atomic functions
     */
    template <typename T>
    class atomic_cuda {
      public:
        /**
         * Function computing an atomic addition
         * @param var reference to variable where the addition is performed
         * @param val value added to var
         * @return the old value contained in var
         */
        GT_FUNCTION_DEVICE static T atomic_add(T &var, const T val) { return ::atomicAdd(&var, val); }

        /**
         * Function computing an atomic substraction
         * @param var reference to variable where the substracion is performed
         * @param val value added to var
         * @return the old value contained in var
         */
        GT_FUNCTION_DEVICE static T atomic_sub(T &var, const T val) { return ::atomicSub(&var, val); }

        /**
         * Function computing an atomic exchange of value of a variable
         * @param var reference to variable which value is replaced by val
         * @param val value inserted in variable var
         * @return the old value contained in var
         */
        GT_FUNCTION_DEVICE static T atomic_exch(T &var, const T val) { return ::atomicExch(&var, val); }

        /**
         * Function computing an atomic min operation
         * @param var reference used to compute and store the min
         * @param val value used in the min comparison
         * @return the old value contained in var
         */
        GT_FUNCTION_DEVICE static T atomic_min(T &var, const T val) { return ::atomicMin(&var, val); }

        /**
         * Function computing an atomic max operation
         * @param var reference used to compute and store the min
         * @param val value used in the min comparison
         * @return the old value contained in var
         */
        GT_FUNCTION_DEVICE static T atomic_max(T &var, const T val) { return ::atomicMax(&var, val); }
    };

    /**
     * Specialization for float
     * generic implementation for CUDA that provides atomic functions
     */
    template <>
    class atomic_cuda<float> {
      public:
        /**
         * Function computing an atomic addition
         * @param var reference to variable where the addition is performed
         * @param val value added to var
         * @return the old value contained in var
         */
        GT_FUNCTION_DEVICE static float atomic_add(float &var, const float val) { return ::atomicAdd(&var, val); }

        /**
         * Function computing an atomic substraction
         * @param var reference to variable where the substracion is performed
         * @param val value added to var
         * @return the old value contained in var
         */
        GT_FUNCTION_DEVICE static float atomic_sub(float &var, const float val) { return ::atomicAdd(&var, -val); }

        /**
         * Function computing an atomic exchange of value of a variable
         * @param var reference to variable which value is replaced by val
         * @param val value inserted in variable var
         * @return the old value contained in var
         */
        GT_FUNCTION_DEVICE static float atomic_exch(float &var, const float val) { return ::atomicExch(&var, val); }

        /**
         * Function computing an atomic min operation
         * @param var reference used to compute and store the min
         * @param val value used in the min comparison
         * @return the old value contained in var
         */
        GT_FUNCTION_DEVICE static float atomic_min(float &var, const float val) {
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
        GT_FUNCTION_DEVICE static float atomic_max(float &var, const float val) {
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
     * Specialization for doubles of AtomicCUDA that provides atomic functions
     */
    template <>
    class atomic_cuda<double> {
      public:
        /**
         * Function computing an atomic addition
         * @param var reference to variable where the addition is performed
         * @param val value added to var
         * @return the old value contained in var
         */
        GT_FUNCTION_DEVICE static double atomic_add(double &var, const double val) {
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
        GT_FUNCTION_DEVICE static double atomic_sub(double &var, const double val) {
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
         *
         * @param x reference to variable which value is replaced by val
         * @param val value inserted in variable var
         * @return the old value contained in x
         */
        GT_FUNCTION_DEVICE static double atomic_exch(double &x, const double val) {
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
        GT_FUNCTION_DEVICE static double atomic_min(double &var, const double val) {
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
        GT_FUNCTION_DEVICE static double atomic_max(double &var, const double val) {
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

    template <typename T>
    struct get_atomic_helper {
        typedef atomic_cuda<T> type;
    };
    /** @} */
    /** @} */
} // namespace gridtools
