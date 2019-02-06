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
#ifndef GT_TARGET_ITERATING
// DON'T USE #pragma once HERE!!!
#ifndef GT_COMMON_GENERIC_METAFUNCTIONS_FOR_EACH_HPP_
#define GT_COMMON_GENERIC_METAFUNCTIONS_FOR_EACH_HPP_

#include "../host_device.hpp"

#define GT_FILENAME <gridtools/common/generic_metafunctions/for_each.hpp>
#include GT_ITERATE_ON_TARGETS()
#undef GT_FILENAME

#endif
#else

namespace gridtools {

    GT_TARGET_NAMESPACE {
        namespace for_each_detail {
            template <class List>
            struct for_each_impl;

            template <template <class...> class L, class... Ts>
            struct for_each_impl<L<Ts...>> {
                template <class Fun>
                GT_TARGET GT_FORCE_INLINE static void exec(Fun const &fun) {
                    (void)(int[]){((void)fun(Ts{}), 0)...};
                }
            };

            template <class List>
            struct for_each_type_impl;

            template <template <class...> class L, class... Ts>
            struct for_each_type_impl<L<Ts...>> {
                template <class Fun>
                GT_TARGET GT_FORCE_INLINE static void exec(Fun const &fun) {
                    (void)(int[]){((void)fun.template operator()<Ts>(), 0)...};
                }
            };

#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ < 9
            // Specialization for empty loops as nvcc refuses to compile the normal version in device code
            template <template <class...> class L>
            struct for_each_impl<L<>> {
                template <class Fun>
                GT_TARGET GT_FORCE_INLINE static void exec(Fun const &) {}
            };

            template <template <class...> class L>
            struct for_each_type_impl<L<>> {
                template <class Fun>
                GT_TARGET GT_FORCE_INLINE static void exec(Fun const &) {}
            };
#endif
        } // namespace for_each_detail

        /** \ingroup common
            @{
            \ingroup allmeta
            @{
            \defgroup foreach For Each
            @{
        */
        /// Calls fun(T{}) for each element of the type list List.
        template <class List, class Fun>
        GT_TARGET GT_FORCE_INLINE void for_each(Fun const &fun) {
            for_each_detail::for_each_impl<List>::exec(fun);
        };

        ///  Calls fun.template operator<T>() for each element of the type list List.
        ///
        ///  Note the difference between for_each: T is passed only as a template parameter; the operator itself has to
        ///  be a nullary function. This ensures that the object of type T is nor created, nor passed to the function.
        ///  The disadvantage is that the functor can not be a [generic] lambda (in C++14 syntax) and also it limits the
        ///  ability to do operator(). However, if T is not a POD it makes sense to use this for_each flavour. Also
        ///  nvcc8 has problems with the code generation for the regular for_each even if all the types are empty
        ///  structs.
        template <class List, class Fun>
        GT_TARGET GT_FORCE_INLINE void for_each_type(Fun const &fun) {
            for_each_detail::for_each_type_impl<List>::exec(fun);
        };

        /** @} */
        /** @} */
        /** @} */
    }
} // namespace gridtools

#endif
