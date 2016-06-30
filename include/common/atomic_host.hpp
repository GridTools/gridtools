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
