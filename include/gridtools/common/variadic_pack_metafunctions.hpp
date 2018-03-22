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

#include <boost/utility.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/or.hpp>
#include <boost/type_traits.hpp>

#include "host_device.hpp"
#include "error.hpp"
#include "defs.hpp"

namespace gridtools {
    /** \ingroup common
        @{
        \defgroup Variadic Pack Metafunctions
        @{
    */

    /**
     *  @brief helper metafunction class that is used to retrieve a value
     *  at a given index of a variadic pack. (step case)
     *  @tparam Size size of the variadic pack
     */
    template < uint_t Size >
    struct get_value_from_pack_functor {
        template < typename First, typename... Dims >
        GT_FUNCTION static constexpr First apply(uint_t Index, First f, Dims... d) {
            return (Index) ? get_value_from_pack_functor< Size - 1 >::apply(Index - 1, d..., f) : f;
        }
    };

    /// \private
    template <>
    struct get_value_from_pack_functor< 0 > {
        template < typename First, typename... Dims >
        GT_FUNCTION static constexpr First apply(uint_t Index, First f, Dims... d) {
            return (Index) ? f : f;
        }
    };

    /**
     *  @brief constexpr function that returns the n-th value from
     *  a given variadic pack.
     *  @tparam First type of the first variadic pack element
     *  @tparam Rest type of the remaining variadic pack elements
     *  @param v The index of the value that should be returned (starting with 0)
     *  @param f first variadic pack element
     *  @param r variadic pack remainders
     *  @return the value of the queried index
     */
    template < typename First, typename... Rest >
    GT_FUNCTION constexpr First get_value_from_pack(uint_t v, First f, Rest... r) {
        return get_value_from_pack_functor< sizeof...(Rest) >::apply(v, f, r...);
    }

    /**
     *  @brief helper metafunction class that is used to retrieve the index
     *  of a variadic pack element. (step case)
     *  @tparam Size size of the variadic pack
     */
    template < uint_t Size >
    struct get_index_of_element_in_pack_functor {
        template < typename First, typename... Dims >
        GT_FUNCTION static constexpr uint_t apply(uint_t Index, First needle, Dims... d) {
            return (get_value_from_pack(Index, d...) == needle)
                       ? Index
                       : get_index_of_element_in_pack_functor< Size - 1 >::apply(Index + 1, needle, d...);
        }
    };

    /// \private
    template <>
    struct get_index_of_element_in_pack_functor< 0 > {
        template < typename First, typename... Dims >
        GT_FUNCTION static constexpr uint_t apply(uint_t Index, First needle, Dims... d) {
            return error_or_return((get_value_from_pack(Index, d...) == needle), Index, "Element not found");
        }
    };

    /**
     *  @brief constexpr function that returns the index of a value
     *  that is part of a variadic pack or fails if the element does not exist.
     *  @tparam First type of the first variadic pack element
     *  @tparam Rest type of the remaining variadic pack elements
     *  @param start_index the starting index
     *  @param needle The value that should be found
     *  @param r other variadic pack elements
     *  @return the index of the queried element
     */
    template < typename First, typename... Rest >
    GT_FUNCTION constexpr uint_t get_index_of_element_in_pack(uint_t start_index, First needle, Rest... r) {
        return get_index_of_element_in_pack_functor< sizeof...(Rest) >::apply(start_index, needle, r...);
    }

    /// \private
    template < typename First >
    GT_FUNCTION constexpr bool is_continuous(First first) {
        return true;
    }

    /*
     * @brief returns true if values are a contiguous sequence of integers
     * @param first Variadic list of integral values. Requires at least two values
     *
     * \param first First mandatory argument
     * \param second Second mandatory argument
     * \param rest Rest of values
     */
    template < typename First, typename Second, typename... Rest >
    GT_FUNCTION constexpr bool is_continuous(First first, Second second, Rest... rest) {
        return (first + 1 == second) ? (true && is_continuous(second, rest...)) : false;
    }
    /** @} */
    /** @} */
}
