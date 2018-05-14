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
            struct check_extent_f {
                template < class N >
                void operator()(N, const gt_fortran_array_descriptor &descriptor) const {
                    if (std::extent< T, N::value >::value != descriptor.dims[descriptor.rank - N::value - 1])
                        throw std::runtime_error("Extents do not match");
                }
            };
        }

        template < class, class = void >
        struct is_fortran_array_view : std::false_type {};

        template < class T >
        struct is_fortran_array_view< T,
            enable_if_t< std::is_same< decay_t< T >, gt_fortran_array_descriptor >::value ||
                                          std::is_convertible< gt_fortran_array_descriptor, T >::value > >
            : std::true_type {};

        template < class T >
        struct is_fortran_array_view<
            T,
            enable_if_t< std::is_lvalue_reference< T >::value && std::is_array< remove_reference_t< T > >::value &&
                         std::is_arithmetic< remove_all_extents_t< remove_reference_t< T > > >::value > >
            : std::true_type {};

        template < class T >
        struct is_fortran_array_view< T,
            enable_if_t< std::is_same< decltype(gt_make_fortran_array_view(
                                           std::declval< gt_fortran_array_descriptor * >(), std::declval< T * >())),
                T >::value > > : std::true_type {};

        template < class T >
        enable_if_t< std::is_same< decay_t< T >, gt_fortran_array_descriptor >::value ||
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
            if (descriptor.rank != std::rank< remove_reference_t< T > >()) {
                throw std::runtime_error("Rank does not match");
            }
            using indices = meta::make_indices< std::rank< remove_reference_t< T > >::value >;
            for_each< indices >(std::bind(
                _impl::check_extent_f< remove_reference_t< T > >{}, std::placeholders::_1, std::cref(descriptor)));

            return reinterpret_cast< T >(descriptor.data);
        }
        template < class T >
        enable_if_t< std::is_same< decltype(gt_make_fortran_array_view(
                                       std::declval< gt_fortran_array_descriptor * >(), std::declval< T * >())),
                         T >::value,
            T >
        make_fortran_array_view(gt_fortran_array_descriptor &descriptor) {
            return gt_make_fortran_array_view(&descriptor, static_cast< T * >(nullptr));
        }

        template < class T, class = void >
        struct fortran_array_view_element_type {
            using type = typename T::gt_view_element_type;
        };
        template < class T >
        struct fortran_array_view_element_type<
            T,
            enable_if_t< std::is_lvalue_reference< T >::value && std::is_array< remove_reference_t< T > >::value &&
                         std::is_arithmetic< remove_all_extents_t< remove_reference_t< T > > >::value > >
            : std::remove_all_extents< remove_reference_t< T > > {};

        template < class T, class = void >
        struct fortran_array_view_rank {
            using type = typename T::gt_view_rank;
        };
        template < class T >
        struct fortran_array_view_rank<
            T,
            enable_if_t< std::is_lvalue_reference< T >::value && std::is_array< remove_reference_t< T > >::value &&
                         std::is_arithmetic< remove_all_extents_t< remove_reference_t< T > > >::value > >
            : std::rank< remove_reference_t< T > > {};
    }
}
