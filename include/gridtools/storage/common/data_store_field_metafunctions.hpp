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

#include <array>
#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include <boost/mpl/at.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/utility.hpp>

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"

namespace gridtools {

    /** \ingroup storage
     * @{
     */

    namespace _impl {
        /** Helper struct for initializing std arrays (used in initializer lists) in a constexpr way */
        template <std::size_t N, typename T, typename... Ts>
        struct array_emplacer {
            template <typename... Args,
                typename = typename boost::enable_if_c<(int)(sizeof...(Args), sizeof...(Ts) + 1 < N), void>::type>
            static constexpr std::array<T, N> emplace(Args const &... args) {
                return array_emplacer<N, T, T, Ts...>::emplace(args...);
            }

            template <typename... Args,
                typename = typename boost::enable_if_c<(int)(sizeof...(Args), sizeof...(Ts) + 1 == N), void>::type,
                typename = void>
            static constexpr std::array<T, N> emplace(Args const &... args) {
                return std::array<T, N>{T(args...), Ts(args...)...};
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
    template <typename T, std::size_t N, typename... Args>
    constexpr std::array<T, N> fill_array(Args const &... args) {
        return _impl::array_emplacer<N, T>::emplace(args...);
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
    template <typename First, typename... Ints>
    GT_FUNCTION constexpr uint_t get_accumulated_data_field_index(int N, First F, Ints... M) {
        return (N == 0) ? 0 : F + get_accumulated_data_field_index(N - 1, M...);
    }

    /**
     * @}
     */
} // namespace gridtools
