/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include <algorithm>
namespace gridtools {

    template < typename T >
    class atomic_host {

      public:
        /**
        * Function computing an atomic addition
        * @param var reference to variable where the addition is performed
        * @param val value added to var
        * @return the old value contained in var
        */
        GT_FUNCTION
        static T atomic_add(T &var, const T val) {
#if _OPENMP > 201106
            T old;
#pragma omp atomic capture
            {
                old = var;
                var += val;
            }
            return old;
#else
            T old;
#pragma omp critical(AtomicAdd)
            {
                old = var;
                var += val;
            }
            return old;
#endif
        }

        /**
        * Function computing an atomic substraction
        * @param var reference to variable where the substraction is performed
        * @param val value added to var
        * @return the old value contained in var
        */
        GT_FUNCTION
        static T atomic_sub(T &var, const T val) {
#if _OPENMP > 201106
            T old;
#pragma omp atomic capture
            {
                old = var;
                var -= val;
            }
            return old;
#else
            T old;
#pragma omp critical(AdtomicSub)
            {
                old = var;
                var -= val;
            }
            return old;
#endif
        }

        /**
        * Function computing an atomic exchange of value of a variable
        * @param var reference to variable which value is replaced by val
        * @param val value inserted in variable var
        * @return the old value contained in var
        */
        GT_FUNCTION
        static T atomic_exch(T &var, const T val) {
#if _OPENMP > 201106
            T old;
#pragma omp capture
            {
                old = var;
                var = val;
            }
            return old;
#else
            T old;
#pragma omp critical(exch)
            {
                old = var;
                var = val;
            }
            return old;
#endif
        }

        /**
        * Function computing an atomic min operation
        * @param var reference used to compute and store the min
        * @param val value used in the min comparison
        * @return the old value contained in var
        */
        GT_FUNCTION
        static T atomic_min(T &var, const T val) {
            T old;
#pragma omp critical(min)
            {
                old = var;
                var = std::min(var, val);
            }
            return old;
        }

        /**
        * Function computing an atomic max operation
        * @param var reference used to compute and store the min
        * @param val value used in the min comparison
        * @return the old value contained in var
        */
        GT_FUNCTION
        static T atomic_max(T &var, const T val) {
            T old;
#pragma omp critical(max)
            {
                old = var;
                var = std::max(var, val);
            }
            return old;
        }
    };

    template < typename T >
    struct get_atomic_helper {
        typedef atomic_host< T > type;
    };

} // namespace gridtools
