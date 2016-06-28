#pragma once
#include "host_device.hpp"

#include <cmath>
#ifdef __CUDACC__
#include "common/atomic_cuda.hpp"
#else
#include "common/atomic_host.hpp"
#endif

/**
* Namespace providing a set of atomic functions working for all backends
*/
namespace gridtools {
    /**
    * Function computing an atomic addition
    * @param var reference to variable where the addition is performed
    * @param val value added to var
    * @return the old value contained in var
    */
    template < typename T >
    GT_FUNCTION T atomic_add(T &var, const T val) {
        return get_atomic_helper< T >::type::atomic_add(var, val);
    }

    /**
    * Function computing an atomic substraction
    * @param var reference to variable where the substraction is performed
    * @param val value added to var
    * @return the old value contained in var
    */
    template < typename T >
    GT_FUNCTION T atomic_sub(T &var, const T val) {
        return get_atomic_helper< T >::type::atomic_sub(var, val);
    }

    /**
    * Function computing an atomic exchange
    * @param var reference to variable which value is replaced by val
    * @param val value inserted in variable var
    * @return the old value contained in var
    */
    template < typename T >
    GT_FUNCTION T atomic_exch(T &var, const T val) {
        return get_atomic_helper< T >::type::atomic_exch(var, val);
    }

    /**
    * Function computing an atomic min operation
    * @param var reference used to compute and store the min
    * @param val value used in the min comparison
    * @return the old value contained in var
    */
    template < typename T >
    GT_FUNCTION T atomic_min(T &var, const T val) {
        return get_atomic_helper< T >::type::atomic_min(var, val);
    }

    /**
    * Function computing an atomic max operation
    * @param var reference used to compute and store the min
    * @param val value used in the min comparison
    * @return the old value contained in var
    */
    template < typename T >
    GT_FUNCTION T atomic_max(T &var, const T val) {
        return get_atomic_helper< T >::type::atomic_max(var, val);
    }
}
