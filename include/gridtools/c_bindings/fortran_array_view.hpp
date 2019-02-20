/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../common/generic_metafunctions/for_each.hpp"
#include "../meta/macros.hpp"
#include "../meta/make_indices.hpp"
#include "../meta/type_traits.hpp"

#include "array_descriptor.h"

namespace gridtools {
    namespace c_bindings {
        namespace _impl {
            template <class T>
            struct fill_extent_f {
                template <class N>
                void operator()(N, gt_fortran_array_descriptor &descriptor) const {
                    descriptor.dims[N::value] = std::extent<T, N::value>::value;
                }
            };
            template <class fortran_type>
            struct fortran_array_element_kind_impl;
            template <>
            struct fortran_array_element_kind_impl<bool> : std::integral_constant<gt_fortran_array_kind, gt_fk_Bool> {};
            template <>
            struct fortran_array_element_kind_impl<int> : std::integral_constant<gt_fortran_array_kind, gt_fk_Int> {};
            template <>
            struct fortran_array_element_kind_impl<short> : std::integral_constant<gt_fortran_array_kind, gt_fk_Short> {
            };
            template <>
            struct fortran_array_element_kind_impl<long> : std::integral_constant<gt_fortran_array_kind, gt_fk_Long> {};
            template <>
            struct fortran_array_element_kind_impl<long long>
                : std::integral_constant<gt_fortran_array_kind, gt_fk_LongLong> {};
            template <>
            struct fortran_array_element_kind_impl<float> : std::integral_constant<gt_fortran_array_kind, gt_fk_Float> {
            };
            template <>
            struct fortran_array_element_kind_impl<double>
                : std::integral_constant<gt_fortran_array_kind, gt_fk_Double> {};
            template <>
            struct fortran_array_element_kind_impl<long double>
                : std::integral_constant<gt_fortran_array_kind, gt_fk_LongDouble> {};
            template <>
            struct fortran_array_element_kind_impl<signed char>
                : std::integral_constant<gt_fortran_array_kind, gt_fk_SignedChar> {};
        } // namespace _impl

        template <class, class = void>
        struct fortran_array_element_kind;
        template <class T>
        struct fortran_array_element_kind<T, enable_if_t<std::is_integral<T>::value>>
            : _impl::fortran_array_element_kind_impl<typename std::make_signed<T>::type> {};
        template <class T>
        struct fortran_array_element_kind<T, enable_if_t<std::is_floating_point<T>::value>>
            : _impl::fortran_array_element_kind_impl<T> {};

        namespace get_fortran_view_meta_impl {
            template <class T, class Arr = remove_reference_t<T>, class ElementType = remove_all_extents_t<Arr>>
            enable_if_t<std::is_array<Arr>::value && std::is_arithmetic<ElementType>::value,
                gt_fortran_array_descriptor>
            get_fortran_view_meta(T *) {
                gt_fortran_array_descriptor descriptor;
                descriptor.type = fortran_array_element_kind<ElementType>::value;
                descriptor.rank = std::rank<Arr>::value;
                descriptor.is_acc_present = false;

                using indices = GT_META_CALL(meta::make_indices, std::rank<Arr>);
                host::for_each<indices>(
                    std::bind(_impl::fill_extent_f<Arr>{}, std::placeholders::_1, std::ref(descriptor)));

                return descriptor;
            }

            template <class T>
            enable_if_t<(T::gt_view_rank::value > 0) && std::is_arithmetic<typename T::gt_view_element_type>::value &&
                            (T::gt_is_acc_present::value == T::gt_is_acc_present::value),
                gt_fortran_array_descriptor>
            get_fortran_view_meta(T *) {
                gt_fortran_array_descriptor descriptor;
                descriptor.type = fortran_array_element_kind<typename T::gt_view_element_type>::value;
                descriptor.rank = T::gt_view_rank::value;
                descriptor.is_acc_present = T::gt_is_acc_present::value;

                return descriptor;
            }
        } // namespace get_fortran_view_meta_impl
        using get_fortran_view_meta_impl::get_fortran_view_meta;
        /**
         * A type T is fortran_array_view_inspectable, one of the following conditions holds:
         *
         * - There exists a function
         *
         *   @code
         *   gt_fortran_array_descriptor get_fortran_view_meta(T*)
         *   @endcode
         *
         *   which returns the meta-data of the type `T`. type, rank and is_acc_present must be set correctly.
         *
         * - T defines T::gt_view_element_type as the element type of the array, T::gt_view_rank is an integral
         *   constant holding the rank of the type, and T::gt_is_acc_present is a bool_constant indicating whether
         *   the data is present on device (when compiling with OpenACC, this will pass a device pointer to the
         *   constructor).
         *
         * - T is a reference to a c-array.
         */
        template <class, class = void>
        struct is_fortran_array_view_inspectable : std::false_type {};
        template <class T>
        struct is_fortran_array_view_inspectable<T,
            enable_if_t<std::is_same<decltype(get_fortran_view_meta(std::declval<add_pointer_t<T>>())),
                gt_fortran_array_descriptor>::value>> : std::true_type {};

        /**
         * The concept of fortran_array_convertible requires that a fortran array described by a
         * gt_fortran_array_descriptor can be converted into T:
         *
         * - T is fortran_array_convertible, if T is a reference to an array of a fortran-compatible type (arithmetic
         *   types).
         * - T is fortran_array_convertible, if gt_fortran_array_descriptor is implicity convertible to T
         * - T is fortran_array_convertible, if there exists a function with the following signature:
         *   @code
         *   T gt_make_fortran_array_view(gt_fortran_array_descriptor*, T*)
         *   @endcode
         * .
         */
        template <class, class = void>
        struct is_fortran_array_convertible : std::false_type {};

        template <class T>
        struct is_fortran_array_convertible<T,
            enable_if_t<std::is_same<decay_t<T>, gt_fortran_array_descriptor>::value ||
                        std::is_convertible<gt_fortran_array_descriptor, T>::value>> : std::true_type {};

        template <class T>
        struct is_fortran_array_convertible<T,
            enable_if_t<std::is_lvalue_reference<T>::value && std::is_array<remove_reference_t<T>>::value &&
                        std::is_arithmetic<remove_all_extents_t<remove_reference_t<T>>>::value>> : std::true_type {};

        template <class T>
        struct is_fortran_array_convertible<T,
            enable_if_t<std::is_same<decltype(gt_make_fortran_array_view(
                                         std::declval<gt_fortran_array_descriptor *>(), std::declval<T *>())),
                T>::value>> : std::true_type {};

        /**
         * @brief A type is fortran_array_bindable if it is fortran_array_convertible
         *
         * A fortran_array_bindable type will appear in the c-bindings as a gt_fortran_array_descriptor.
         */
        template <class T>
        struct is_fortran_array_bindable : is_fortran_array_convertible<T> {};
        /**
         * @brief A type is fortran_array_wrappable if it is both fortran_array_bindable and
         * fortran_array_view_inspectable.
         *
         * If used with the wrapper-versions of the export-function, fortran_array_wrappable types can be created from a
         * fortran array in the fortran bindings, whereas fortran_array_convertible-types that are not bindable will
         * appear as gt_fortran_array_descriptors and must be filled manually.
         */
        template <class T>
        struct is_fortran_array_wrappable
            : bool_constant<is_fortran_array_bindable<T>::value && is_fortran_array_view_inspectable<T>::value> {};

        template <class T>
        enable_if_t<std::is_same<decay_t<T>, gt_fortran_array_descriptor>::value ||
                        std::is_convertible<gt_fortran_array_descriptor, T>::value,
            T>
        make_fortran_array_view(gt_fortran_array_descriptor *descriptor) {
            return *descriptor;
        }
        template <class T>
        enable_if_t<std::is_lvalue_reference<T>::value && std::is_array<remove_reference_t<T>>::value &&
                        std::is_arithmetic<remove_all_extents_t<remove_reference_t<T>>>::value,
            T>
        make_fortran_array_view(gt_fortran_array_descriptor *descriptor) {
            static gt_fortran_array_descriptor cpp_meta = get_fortran_view_meta((add_pointer_t<T>){nullptr});
            if (descriptor->type != cpp_meta.type) {
                throw std::runtime_error("Types do not match: fortran-type (" + std::to_string(descriptor->type) +
                                         ") != c-type (" + std::to_string(cpp_meta.type) + ")");
            }
            if (descriptor->rank != cpp_meta.rank) {
                throw std::runtime_error("Rank does not match: fortran-rank (" + std::to_string(descriptor->rank) +
                                         ") != c-rank (" + std::to_string(cpp_meta.rank) + ")");
            }
            for (int i = 0; i < descriptor->rank; ++i) {
                if (cpp_meta.dims[i] != descriptor->dims[descriptor->rank - i - 1])
                    throw std::runtime_error("Extents do not match");
            }

            return *reinterpret_cast<remove_reference_t<T> *>(descriptor->data);
        }
        template <class T>
        enable_if_t<std::is_same<decltype(gt_make_fortran_array_view(
                                     std::declval<gt_fortran_array_descriptor *>(), std::declval<T *>())),
                        T>::value,
            T>
        make_fortran_array_view(gt_fortran_array_descriptor *descriptor) {
            return gt_make_fortran_array_view(descriptor, (T *){nullptr});
        }
    } // namespace c_bindings
} // namespace gridtools
