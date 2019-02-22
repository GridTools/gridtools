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

#include <type_traits>

#include <stdbool.h>

#include "../common/any_moveable.hpp"

#include "fortran_array_view.hpp"
#include "handle_impl.hpp"

namespace gridtools {
    namespace c_bindings {
        namespace _impl {

            template <class T, class = void>
            struct result_converted_to_c;

            template <class T>
            struct result_converted_to_c<T,
                typename std::enable_if<std::is_void<T>::value || std::is_arithmetic<T>::value>::type> {
                using type = T;
            };

            template <class T>
            struct result_converted_to_c<T,
                typename std::enable_if<std::is_class<typename std::remove_reference<T>::type>::value>::type> {
                using type = gt_handle *;
            };

            template <class T, class = void>
            struct param_converted_to_c;

            template <class T>
            struct param_converted_to_c<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
                using type = T;
            };

            template <class T>
            struct param_converted_to_c<T *, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
                using type = T *;
            };

            template <class T>
            struct param_converted_to_c<T &, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
                using type = T *;
            };

            template <class T>
            struct param_converted_to_c<T *,
                typename std::enable_if<std::is_class<T>::value && !is_fortran_array_bindable<T *>::value>::type> {
                using type = gt_handle *;
            };
            template <class T>
            struct param_converted_to_c<T,
                typename std::enable_if<std::is_class<remove_reference_t<T>>::value &&
                                        !is_fortran_array_bindable<T>::value>::type> {
                using type = gt_handle *;
            };

            template <class T>
            struct param_converted_to_c<T, typename std::enable_if<is_fortran_array_bindable<T>::value>::type> {
                using type = gt_fortran_array_descriptor *;
            };

            template <class T, typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
            T convert_to_c(T obj) {
                return obj;
            }

            template <class T,
                typename std::enable_if<std::is_class<typename std::remove_reference<T>::type>::value, int>::type = 0>
            gt_handle *convert_to_c(T &&obj) {
                return new gt_handle{std::forward<T>(obj)};
            }

            template <class T>
            using result_converted_to_c_t = typename result_converted_to_c<T>::type;
            template <class T>
            using param_converted_to_c_t = typename param_converted_to_c<T>::type;

            template <class T,
                typename std::enable_if<std::is_arithmetic<typename std::remove_pointer<T>::type>::value, int>::type =
                    0>
            T convert_from_c(T obj) {
                return obj;
            };

            template <class T,
                typename std::enable_if<std::is_reference<T>::value &&
                                            std::is_arithmetic<typename std::remove_reference<T>::type>::value,
                    int>::type = 0>
            T convert_from_c(typename std::remove_reference<T>::type *obj) {
                return *obj;
            };

            template <class T, typename std::enable_if<std::is_pointer<T>::value, int>::type = 0>
            T convert_from_c(gt_handle *obj) {
                return &any_cast<remove_pointer_t<T> &>(obj->m_value);
            }
            template <class T, typename std::enable_if<!std::is_pointer<T>::value, int>::type = 0>
            T convert_from_c(gt_handle *obj) {
                return any_cast<T>(obj->m_value);
            }
            template <class T>
            T convert_from_c(gt_fortran_array_descriptor *obj) {
                return make_fortran_array_view<T>(obj);
            }

            template <class T, class Impl>
            struct wrapped_f;

            template <class R, class... Params, class Impl>
            struct wrapped_f<R(Params...), Impl> {
                Impl m_fun;
                result_converted_to_c_t<R> operator()(param_converted_to_c_t<Params>... args) const {
                    return convert_to_c(m_fun(convert_from_c<Params>(args)...));
                }
            };

            template <class... Params, class Impl>
            struct wrapped_f<void(Params...), Impl> {
                Impl m_fun;
                void operator()(param_converted_to_c_t<Params>... args) const {
                    m_fun(convert_from_c<Params>(args)...);
                }
            };

            template <class T>
            struct wrapped;

            template <class T>
            struct wrapped<T *> {
                using type = typename wrapped<T>::type;
            };

            template <class T>
            struct wrapped<T &> {
                using type = typename wrapped<T>::type;
            };

            template <class R, class... Params>
            struct wrapped<R(Params...)> {
                using type = result_converted_to_c_t<R>(typename param_converted_to_c<Params>::type...);
            };
        } // namespace _impl

        /// Transform a function type to to the function type that is callable from C
        template <class T>
        using wrapped_t = typename _impl::wrapped<T>::type;

        /// Wrap the functor of type `Impl` to another functor that can be invoked with the 'wrapped_t<T>' signature.
        template <class T, class Impl>
        constexpr _impl::wrapped_f<T, typename std::decay<Impl>::type> wrap(Impl &&obj) {
            return {std::forward<Impl>(obj)};
        }

        /// Specialization for function pointers.
        template <class T>
        constexpr _impl::wrapped_f<T, T *> wrap(T *obj) {
            return {obj};
        }
    } // namespace c_bindings
} // namespace gridtools
