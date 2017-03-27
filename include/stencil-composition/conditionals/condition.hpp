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
#include "conditional.hpp"
/**@file*/
namespace gridtools {

    /**@brief structure containing a conditional and the two branches

       This structure is the record associated to a conditional, it contains two multi-stage stencils,
       possibly containing other conditionals themselves. One branch or the other will be eventually
       executed, depending on the content of the m_value member variable.
     */
    template < typename Mss1, typename Mss2, typename Tag >
    struct condition {

        GRIDTOOLS_STATIC_ASSERT(!(is_reduction_descriptor< Mss1 >::value || is_reduction_descriptor< Mss2 >::value),
            "Reduction multistage must be outside conditional branches");
        // TODO add a way to check Mss1 and Mss2
        GRIDTOOLS_STATIC_ASSERT(is_conditional< Tag >::value, GT_INTERNAL_ERROR);
        typedef Mss1 first_t;
        typedef Mss2 second_t;
        typedef Tag index_t;

      private:
        index_t m_value;
        first_t m_first;
        second_t m_second;

      public:
        constexpr condition(){};

        constexpr condition(index_t cond, first_t const &first_, second_t const &second_)
            : m_value(cond), m_first(first_), m_second(second_) {}

        constexpr index_t value() const { return m_value; }
        constexpr second_t const &second() const { return m_second; }
        constexpr first_t const &first() const { return m_first; }
    };

    template < typename T >
    struct is_condition : boost::mpl::false_ {};

    template < typename Mss1, typename Mss2, typename Tag >
    struct is_condition< condition< Mss1, Mss2, Tag > > : boost::mpl::true_ {};
}
