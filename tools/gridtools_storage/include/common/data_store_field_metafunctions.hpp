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

#include <array>
#include <type_traits>

#include <boost/mpl/at.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/utility.hpp>

namespace gridtools {

    /* Helper struct for initializing an std arrays */
    template < std::size_t N, typename T, typename... Ts >
    struct array_emplacer {
        template < typename... Args,
            typename = typename boost::enable_if_c< (int)(sizeof...(Args), sizeof...(Ts) + 1 < N), void >::type >
        static std::array< T, N > emplace(Args const &... args) {
            return array_emplacer< N, T, T, Ts... >::emplace(args...);
        }

        template < typename... Args,
            typename = typename boost::enable_if_c< (int)(sizeof...(Args), sizeof...(Ts) + 1 == N), void >::type,
            typename = void >
        static std::array< T, N > emplace(Args const &... args) {
            return std::array< T, N >{T(args...), Ts(args...)...};
        }
    };

    template < typename T, std::size_t N, typename... Args >
    std::array< T, N > emplace_array(Args const &... args) {
        return array_emplacer< N, T >::emplace(args...);
    }

    /* Helper struct for generating the correct initializer list when given #Components different StorageInfos */

    // pass an mpl::vector< int<2>, int<2>, int<1>, int<0>, int<0> >
    // and the output value will be an std::array that is filled with {2,2,1,0,0}
    template < typename Vector, unsigned Cnt = boost::mpl::size< Vector >::value, unsigned... N >
    struct get_vals : get_vals< Vector, Cnt - 1, boost::mpl::at_c< Vector, sizeof...(N) >::type::value, N... > {};

    template < typename Vector, unsigned... N >
    struct get_vals< Vector, 0, N... > {
        constexpr static std::array< unsigned, sizeof...(N) > value = {N...};

        // method that can be used to do a component wise initialization of a data_field (with different storage infos)
        template < typename T, typename V, typename... Values >
        static std::array< T, sizeof...(N) > generator(Values... v) {
            return {T(std::array< V, sizeof...(Values) >({v...})[sizeof...(Values)-1 - N])...};
        }
    };

    // pass a start type (e.g, mpl::vector<>) and pass the numbers (e.g., 2,1,2)
    // the metafunction here will return a sequence that counts each component down to 0
    // and returns e.g., mpl::vector< int<2>, int<2>, int<1>, int<0>, int<0> >
    template < typename Start, unsigned First, unsigned... N >
    struct get_sequence {
        typedef typename get_sequence< Start, First - 1, N... >::type pre_vec;
        typedef typename boost::mpl::push_back< pre_vec, boost::mpl::int_< sizeof...(N) > >::type type;
    };

    template < typename Start, unsigned... N >
    struct get_sequence< Start, 1, N... > {
        typedef typename get_sequence< Start, N... >::type pre_vec;
        typedef typename boost::mpl::push_back< pre_vec, boost::mpl::int_< sizeof...(N) > >::type type;
    };

    template < typename Start >
    struct get_sequence< Start, 1 > {
        typedef typename boost::mpl::push_back< Start, boost::mpl::int_< 0 > >::type type;
    };

    /* Given a data_field<T, MetaData, X...> this function will accumulate X... until a given point (N). */
    constexpr unsigned get_accumulated_data_field_index(int N) { return 0; }

    template < typename First, typename... Ints >
    constexpr unsigned get_accumulated_data_field_index(int N, First F, Ints... M) {
        return (N == 0) ? 0 : F + get_accumulated_data_field_index(N - 1, M...);
    }
}
