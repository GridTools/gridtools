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

/**@file
   @brief contains the API of the caonditionals type, to be used for specifying the control flow
   in the computation tree.

   The user wanting to select a multi-stage stencil at runtime, based on a boolean condition, must instantiate
   this class with a unique ID as template argument, construct it using the boolean condition, and then
   use the \ref gridtools::if_ statement from whithin the make_computation.
*/
#include <functional>

namespace gridtools {
    struct condition_functor {
        std::function< short_t() > m_1;
        short_t m_2;
        condition_functor(std::function< int() > t1_, short_t t2_) : m_1(t1_), m_2(t2_) {}
        condition_functor(std::function< bool() > t1_) : m_1([t1_]() -> short_t { return t1_() ? 0 : 1; }), m_2(0) {}

        condition_functor() : m_1([]() { return 0; }), m_2(0) {}

        bool operator()() const { return m_1() == m_2; }
    };
}

#define BOOL_FUNC(val) std::function< bool() > val

namespace gridtools {

    template < uint_t Tag, uint_t SwitchId = 0 >
    class conditional {

        // weak pointer, viewing the boolean condition
        std::function< bool() > m_value;

      public:
        typedef static_uint< Tag > index_t;
        static const uint_t index_value = index_t::value;

        /**
           @brief default constructor
         */
        conditional() // try to avoid this?
            : m_value([]() {
                assert(false);
                return false;
            }) {}

        /**
           @brief constructor for switch variables (for GCC53 bug)

           This constructor should not be needed
        */
        conditional(BOOL_FUNC(c)) : m_value(c) {}

        /**@brief returns the boolean condition*/
        bool value() const { return m_value(); }
    };

    template < typename T >
    struct is_conditional : boost::mpl::false_ {};

    template < uint_t Tag >
    struct is_conditional< conditional< Tag > > : boost::mpl::true_ {};

    template < uint_t Tag, uint_t SwitchId >
    struct is_conditional< conditional< Tag, SwitchId > > : boost::mpl::true_ {};
}
