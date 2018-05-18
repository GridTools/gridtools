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
#include <stdexcept>
#include "../common/generic_metafunctions/type_traits.hpp"
#include "../common/generic_metafunctions/meta.hpp"
#include "../common/generic_metafunctions/for_each.hpp"

#include "array_descriptor.h"

namespace gridtools {
    namespace c_bindings {
        namespace _impl {
            template < class T >
            struct fill_extent_f {
                template < class N >
                void operator()(N, gt_fortran_array_descriptor &descriptor) const {
                    descriptor.dims[N::value] = std::extent< T, N::value >::value;
                }
            };
            template < class fortran_type >
            struct fortran_array_element_kind_impl;
            template <>
            struct fortran_array_element_kind_impl< bool > {
                static constexpr gt_fortran_array_kind value = gt_fortran_array_kind::gt_fk_Bool;
            };
            template <>
            struct fortran_array_element_kind_impl< int > {
                static constexpr gt_fortran_array_kind value = gt_fortran_array_kind::gt_fk_Int;
            };
            template <>
            struct fortran_array_element_kind_impl< short > {
                static constexpr gt_fortran_array_kind value = gt_fortran_array_kind::gt_fk_Short;
            };
            template <>
            struct fortran_array_element_kind_impl< long > {
                static constexpr gt_fortran_array_kind value = gt_fortran_array_kind::gt_fk_Long;
            };
            template <>
            struct fortran_array_element_kind_impl< long long > {
                static constexpr gt_fortran_array_kind value = gt_fortran_array_kind::gt_fk_LongLong;
            };
            template <>
            struct fortran_array_element_kind_impl< float > {
                static constexpr gt_fortran_array_kind value = gt_fortran_array_kind::gt_fk_Float;
            };
            template <>
            struct fortran_array_element_kind_impl< double > {
                static constexpr gt_fortran_array_kind value = gt_fortran_array_kind::gt_fk_Double;
            };
            template <>
            struct fortran_array_element_kind_impl< long double > {
                static constexpr gt_fortran_array_kind value = gt_fortran_array_kind::gt_fk_LongDouble;
            };
            template <>
            struct fortran_array_element_kind_impl< signed char > {
                static constexpr gt_fortran_array_kind value = gt_fortran_array_kind::gt_fk_SignedChar;
            };

            template < class, class = void >
            struct fortran_array_element_kind;
            template < class T >
            struct fortran_array_element_kind< T, enable_if_t< std::is_integral< T >::value > > {
                static constexpr gt_fortran_array_kind value =
                    fortran_array_element_kind_impl< typename std::make_signed< T >::type >::value;
            };
            template < class T >
            struct fortran_array_element_kind< T, enable_if_t< std::is_floating_point< T >::value > > {
                static constexpr gt_fortran_array_kind value = fortran_array_element_kind_impl< T >::value;
            };
        }

        namespace get_fortran_view_meta_impl {
            template < class T >
            enable_if_t< std::is_array< remove_reference_t< T > >::value &&
                             std::is_arithmetic< remove_all_extents_t< remove_reference_t< T > > >::value,
                gt_fortran_array_descriptor >
            get_fortran_view_meta(T *) {
                gt_fortran_array_descriptor descriptor;
                descriptor.type =
                    _impl::fortran_array_element_kind< remove_all_extents_t< remove_reference_t< T > > >::value;
                descriptor.rank = std::rank< remove_reference_t< T > >::value;

                using indices = meta::make_indices< std::rank< remove_reference_t< T > >::value >;
                for_each< indices >(std::bind(
                    _impl::fill_extent_f< remove_reference_t< T > >{}, std::placeholders::_1, std::ref(descriptor)));

                return descriptor;
            }

            template < class T >
            enable_if_t< (T::gt_view_rank::value > 0) && std::is_arithmetic< typename T::gt_view_element_type >::value,
                gt_fortran_array_descriptor >
            get_fortran_view_meta(T *) {
                gt_fortran_array_descriptor descriptor;
                descriptor.type = _impl::fortran_array_element_kind< typename T::gt_view_element_type >::value;
                descriptor.rank = T::gt_view_rank::value;

                return descriptor;
            }
        }
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
         *   which returns the meta-data of the type `T`. type and rank must be set correctly.
         *
         * - T defines T::gt_view_element_type as the element types of the array and T::gt_view_rank is an integral
         *   constant holding the rank of the type
         *
         * - T is a reference to a c-array.
         */
        template < class, class = void >
        struct is_fortran_array_view_inspectable : std::false_type {};
        template < class T >
        struct is_fortran_array_view_inspectable< T,
            enable_if_t< std::is_same< decltype(get_fortran_view_meta(std::declval< add_pointer_t< T > >())),
                gt_fortran_array_descriptor >::value > > : std::true_type {};

        /**
         * The concept of fortran_array_convertible requires that a fortran array described by a
         * gt_fortran_array_descriptor can be converted into T:
         *
         * - T is fortran_array_convertible, if T is a reference to an array of a fortran-compatible type (arithmetic
         *   types).
         * - T is fortran_array_convertible, if gt_fortran_array_descriptor is implicity convertible to T
         * - T is fortran_array_convertible, if there exists a function with the following signature:
         *
         *   @code
         *   T gt_make_fortran_array_view(gt_fortran_array_descriptor*, T*)
         *   @endcode
         * .
         */
        template < class, class = void >
        struct is_fortran_array_convertible : std::false_type {};

        template < class T >
        struct is_fortran_array_convertible< T,
            enable_if_t< std::is_same< decay_t< T >, gt_fortran_array_descriptor >::value ||
                                                 std::is_convertible< gt_fortran_array_descriptor, T >::value > >
            : std::true_type {};

        template < class T >
        struct is_fortran_array_convertible<
            T,
            enable_if_t< std::is_lvalue_reference< T >::value && std::is_array< remove_reference_t< T > >::value &&
                         std::is_arithmetic< remove_all_extents_t< remove_reference_t< T > > >::value > >
            : std::true_type {};

        template < class T >
        struct is_fortran_array_convertible< T,
            enable_if_t< std::is_same< decltype(gt_make_fortran_array_view(
                                           std::declval< gt_fortran_array_descriptor * >(), std::declval< T * >())),
                T >::value > > : std::true_type {};

        /**
         * @brief A type is fortran_array_bindable if it is fortran_array_convertible
         *
         * A fortran_array_bindable type will appear in the c-bindings as a gt_fortran_array_descriptor.
         */
        template < class T >
        struct is_fortran_array_bindable : is_fortran_array_convertible< T > {};
        /**
         * @brief A type is fortran_array_wrappable if it is both fortran_array_bindable and
         * fortran_array_view_inspectable.
         *
         * If used with the wrapper-versions of the export-function, fortran_array_wrappable types can be created from a
         * fortran array in the fortran bindings, whereas fortran_array_convertible-types that are not bindable will
         * appear as gt_fortran_array_descriptors and must be filled manually.
         */
        template < class T >
        struct is_fortran_array_wrappable
            : std::integral_constant< bool,
                  is_fortran_array_bindable< T >::value && is_fortran_array_view_inspectable< T >::value > {};

        template < class T >
        enable_if_t< !std::is_same< decay_t< T >, gt_fortran_array_descriptor >::value &&
                         std::is_convertible< gt_fortran_array_descriptor, T >::value,
            T >
        make_fortran_array_view(gt_fortran_array_descriptor &descriptor) {
            return descriptor;
        }
        template < class T >
        enable_if_t< std::is_lvalue_reference< T >::value && std::is_array< remove_reference_t< T > >::value &&
                         std::is_arithmetic< remove_all_extents_t< remove_reference_t< T > > >::value,
            T >
        make_fortran_array_view(gt_fortran_array_descriptor &descriptor) {
            const auto cpp_meta = get_fortran_view_meta((add_pointer_t< T >){nullptr});
            if (descriptor.type != cpp_meta.type) {
                throw std::runtime_error("Types do not match: fortran-type (" + std::to_string(descriptor.type) +
                                         ") != c-type (" + std::to_string(cpp_meta.type) + ")");
            }
            if (descriptor.rank != cpp_meta.rank) {
                throw std::runtime_error("Rank does not match: fortran-rank (" + std::to_string(descriptor.rank) +
                                         ") != c-rank (" + std::to_string(cpp_meta.rank) + ")");
            }
            for (int i = 0; i < descriptor.rank; ++i) {
                if (cpp_meta.dims[i] != descriptor.dims[descriptor.rank - i - 1])
                    throw std::runtime_error("Extents do not match");
            }

            return *reinterpret_cast< remove_reference_t< T > * >(descriptor.data);
        }
        template < class T >
        enable_if_t< std::is_same< decltype(gt_make_fortran_array_view(
                                       std::declval< gt_fortran_array_descriptor * >(), std::declval< T * >())),
                         T >::value,
            T >
        make_fortran_array_view(gt_fortran_array_descriptor &descriptor) {
            return gt_make_fortran_array_view(&descriptor, (T *){nullptr});
        }
    }
}
