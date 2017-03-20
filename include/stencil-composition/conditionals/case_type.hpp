/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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
/**@file*/
namespace gridtools {

    /**@brief construct for storing a case in a @ref gridtools::switch_ statement

       It stores a runtime value associated to the branch, which has to be compared with the value in a
       @ref gridtools::switch_variable, and the corresponding multi-stage stencil
       to be executed in case the condition holds.
     */
    template < typename T, typename Mss >
    struct case_type {
      private:
        T m_value;
        Mss m_mss;

      public:
        case_type(T val_, Mss mss_) : m_value(val_), m_mss(mss_) {}

        Mss mss() const { return m_mss; }
        T value() const { return m_value; }
    };

    template < typename Mss >
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

    template < typename T >
    struct is_case_type : boost::mpl::false_ {};

    template < typename T, typename Mss >
    struct is_case_type< case_type< T, Mss > > : boost::mpl::true_ {};

    template < typename T >
    struct is_default_type : boost::mpl::false_ {};

    template < typename Mss >
    struct is_default_type< default_type< Mss > > : boost::mpl::true_ {};
} // namespace gridtools
