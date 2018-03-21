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

#include <boost/mpl/bool.hpp>

#include "variadic_pack_metafunctions.hpp"
#include "defs.hpp"
#include "gt_assert.hpp"
#include "generic_metafunctions/accumulate.hpp"
#include "generic_metafunctions/type_traits.hpp"

#if GT_BROKEN_TEMPLATE_ALIASES
#include <boost/mpl/greater_equal.hpp>
#include <boost/mpl/filter_view.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/vector.hpp>
#else
#include "generic_metafunctions/meta.hpp"
#endif

namespace gridtools {

    template < int... Args >
    struct layout_map {
      private:
#if GT_BROKEN_TEMPLATE_ALIASES
        /* list of all arguments */
        using args = boost::mpl::vector< boost::mpl::int_< Args >... >;

        /* list of all unmasked (i.e. non-negative) arguments */
        using unmasked_args =
            boost::mpl::filter_view< args, boost::mpl::greater_equal< boost::mpl::_, boost::mpl::int_< 0 > > >;

        /* sum of all unmasked arguments (only used for assertion below) */
        static constexpr int unmasked_arg_sum = boost::mpl::fold< unmasked_args,
            boost::mpl::int_< 0 >,
            boost::mpl::plus< boost::mpl::_1, boost::mpl::_2 > >::type::value;

      public:
        /** @brief Length of layout map excluding masked dimensions. */
        static constexpr std::size_t unmasked_length = boost::mpl::size< unmasked_args >::type::value;
#else
        /* list of all arguments */
        using args = meta::list< std::integral_constant< int, Args >... >;

        /* helper meta functions */
        template < typename Int >
        using not_negative = meta::bool_constant< (Int::value >= 0) >;
        template < typename A, typename B >
        using integral_plus = std::integral_constant< int, A::value + B::value >;

        /* list of all unmasked (i.e. non-negative) arguments */
        using unmasked_args = meta::apply< meta::filter< not_negative >, args >;

        /* sum of all unmasked arguments (only used for assertion below) */
        static constexpr int unmasked_arg_sum = meta::apply< meta::combine< integral_plus >,
            meta::push_back< unmasked_args, std::integral_constant< int, 0 > > >::value;

      public:
        /** @brief Length of layout map excluding masked dimensions. */
        static constexpr std::size_t unmasked_length = meta::length< unmasked_args >::value;
#endif
        /** @brief Total length of layout map, including masked dimensions. */
        static constexpr std::size_t masked_length = sizeof...(Args);

        GRIDTOOLS_STATIC_ASSERT(sizeof...(Args) > 0, GT_INTERNAL_ERROR_MSG("Zero-dimensional layout makes no sense."));
        GRIDTOOLS_STATIC_ASSERT((unmasked_arg_sum == (unmasked_length * (unmasked_length - 1)) / 2),
            GT_INTERNAL_ERROR_MSG("Layout map args must not contain any holes (e.g., layout_map<3,1,0>)."));

        /** @brief Get the position of the element with value `I` in the layout map. */
        template < int I >
        GT_FUNCTION static constexpr std::size_t find() {
            GRIDTOOLS_STATIC_ASSERT(
                (I >= 0) && (I < unmasked_length), GT_INTERNAL_ERROR_MSG("This index does not exist"));
            // force compile-time evaluation
            return std::integral_constant< std::size_t, find(I) >::value;
        }

        /** @brief Get the position of the element with value `i` in the layout map. */
        GT_FUNCTION static constexpr std::size_t find(int i) { return get_index_of_element_in_pack(0, i, Args...); }

        /** @brief Get the value of the element at position `I` in the layout map. */
        template < std::size_t I >
        GT_FUNCTION static constexpr int at() {
            GRIDTOOLS_STATIC_ASSERT(I < masked_length, GT_INTERNAL_ERROR_MSG("Out of bounds access"));
            // force compile-time evaluation
            return std::integral_constant< int, at(I) >::value;
        }

        /** @brief Get the value of the element at position `I` in the layout map. */
        GT_FUNCTION static constexpr int at(std::size_t i) { return get_value_from_pack(i, Args...); }

        /**
         * @brief Version of `at` that does not check the index bound and return -1 for out of bounds indices.
         * Use the versions with bounds check if applicable.
         */
        template < std::size_t I >
        GT_FUNCTION static constexpr int at_unsafe() {
            return I < masked_length ? at< I >() : -1;
        }

        template < std::size_t I >
        GT_FUNCTION static constexpr int select(int const *dims) {
            return dims[at< I >()];
        }

        /** @brief Get the maximum element value in the layout map. */
        GT_FUNCTION static constexpr int max() { return constexpr_max(Args...); }
    };

    template < typename T >
    struct is_layout_map : boost::mpl::false_ {};

    template < int... Args >
    struct is_layout_map< layout_map< Args... > > : boost::mpl::true_ {};
}
