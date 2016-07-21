/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
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
