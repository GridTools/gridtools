/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>

namespace gridtools {

    /**@brief construct for storing a case in a @ref gridtools::switch_ statement

       It stores a runtime value associated to the branch, which has to be compared with the value in a
       switch_variable, and the corresponding multi-stage stencil
       to be executed in case the condition holds.
     */
    template <typename T, typename Mss>
    struct case_type {
      private:
        T m_value;
        Mss m_mss;

      public:
        case_type(T val_, Mss mss_) : m_value(val_), m_mss(mss_) {}

        Mss mss() const { return m_mss; }
        T value() const { return m_value; }
    };

    template <typename Mss>
    struct default_type {
      private:
        Mss m_mss;

        /**@brief construct for storing the default case in a @ref gridtools::switch_ statement

           It stores a multi-stage stencil
           to be executed in case none of the other conditions holds.
         */
      public:
        typedef Mss mss_t;

        default_type(Mss mss_) : m_mss(mss_) {}

        Mss mss() const { return m_mss; }
    };

    template <typename T>
    struct is_case_type : std::false_type {};

    template <typename T, typename Mss>
    struct is_case_type<case_type<T, Mss>> : std::true_type {};

    template <typename T>
    struct is_default_type : std::false_type {};

    template <typename Mss>
    struct is_default_type<default_type<Mss>> : std::true_type {};
} // namespace gridtools
