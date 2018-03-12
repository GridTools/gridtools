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

#include <array>
#include <type_traits>

#include <boost/mpl/at.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/utility.hpp>

namespace gridtools {

    /** \ingroup storage
     * @{
     */

    namespace _impl {
        /** Helper struct for initializing std arrays (used in initializer lists) in a constexpr way */
        template < std::size_t N, typename T, typename... Ts >
        struct array_emplacer {
            template < typename... Args,
                typename = typename boost::enable_if_c< (int)(sizeof...(Args), sizeof...(Ts) + 1 < N), void >::type >
            static constexpr std::array< T, N > emplace(Args const &... args) {
                return array_emplacer< N, T, T, Ts... >::emplace(args...);
            }

            template < typename... Args,
                typename = typename boost::enable_if_c< (int)(sizeof...(Args), sizeof...(Ts) + 1 == N), void >::type,
                typename = void >
            static constexpr std::array< T, N > emplace(Args const &... args) {
                return std::array< T, N >{T(args...), Ts(args...)...};
            }
        };
    } // namespace _impl

    /** \brief Call the constructor of the element type of a std::array with the arguments provided for each element
     *
     * \tparam T Type of the element of the array
     * \tparam N Size of the array
     * \param args Variadic list of the constructor argument for type T
     * \return A std::array<T,N> with elements initialized with the object constructed with the given arguments
     */
    template < typename T, std::size_t N, typename... Args >
    constexpr std::array< T, N > fill_array(Args const &... args) {
        return _impl::array_emplacer< N, T >::emplace(args...);
    }

    /**
     * @brief Given a data_store_field<T, MetaData, X...> this function will
     * accumulate X... until a given point (N). Base case.
     *
     * @param N accumulate until.
     * @return accumulated data field index.
     */
    GT_FUNCTION
    constexpr uint_t get_accumulated_data_field_index(int N) { return 0; }

    /**
     * @brief Given a data_field<T, MetaData, X...> this function will accumulate X... until a given point (N). Step
     * case.
     * @param N accumulate until
     * @param F size of first data field coordinate.
     * @param M variadic list containing the size of other data field coordinates
     * @return accumulated data field index.
     */
    template < typename First, typename... Ints >
    GT_FUNCTION constexpr uint_t get_accumulated_data_field_index(int N, First F, Ints... M) {
        return (N == 0) ? 0 : F + get_accumulated_data_field_index(N - 1, M...);
    }

    /**
     * @}
     */
}
