/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
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

            template <template <class...> class L>
            struct for_each_impl<L<>> {
                template <class Fun>
                GT_TARGET GT_FORCE_INLINE static void exec(Fun const &) {}
            };

            template <template <class...> class L, class... Ts>
            struct for_each_impl<L<Ts...>> {
                template <class Fun>
                GT_TARGET GT_FORCE_INLINE static void exec(Fun const &fun) {
                    (void)(int[]){((void)fun(Ts{}), 0)...};
                }
            };

            template <class List>
            struct for_each_type_impl;

            template <template <class...> class L>
            struct for_each_type_impl<L<>> {
                template <class Fun>
                GT_TARGET GT_FORCE_INLINE static void exec(Fun const &) {}
            };

            template <template <class...> class L, class... Ts>
            struct for_each_type_impl<L<Ts...>> {
                template <class Fun>
                GT_TARGET GT_FORCE_INLINE static void exec(Fun const &fun) {
                    (void)(int[]){((void)fun.template operator()<Ts>(), 0)...};
                }
            };
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
        }

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
        }

        /** @} */
        /** @} */
        /** @} */
    }
} // namespace gridtools

#endif
