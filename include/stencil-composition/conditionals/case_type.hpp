/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
