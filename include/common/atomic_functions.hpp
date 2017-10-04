/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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
