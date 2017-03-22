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
#include "accessor_fwd.hpp"

namespace gridtools {
#ifdef CUDA8 // (i.e. CXX11_ENABLED for cpu)

    /**@brief same as accessor but mixing run-time offsets with compile-time ones

       When we know beforehand that the dimension which we are querying is
       a compile-time one, we can use the static method get_constexpr() to get the offset.
       Otherwise the method get() checks before among the static dimensions, and if the
       queried dimension is not found it looks up in the dynamic dimensions. Note that this
       lookup is anyway done at compile time, i.e. the get() method returns in constant time.
     */
    template < typename ArgType, typename... Pair >
    struct accessor_mixed : public offset_tuple_mixed< typename ArgType::offset_tuple_t, Pair... > {
        typedef typename ArgType::index_t index_t;
        typedef typename ArgType::base_t base_t;
        typedef typename ArgType::offset_tuple_t offset_tuple_t;
        typedef typename ArgType::extent_t extent_t;

        using super = offset_tuple_mixed< typename ArgType::offset_tuple_t, Pair... >;
        /**inheriting all constructors from offset_tuple*/
        using offset_tuple_mixed< typename ArgType::offset_tuple_t, Pair... >::offset_tuple_mixed;

        GT_FUNCTION
        constexpr const super &offsets() const { return *this; }
    };

    /**
       @brief this struct allows the specification of SOME of the arguments before instantiating the offset_tuple.
       It is a language keyword. Usage examples can be found in the unit test \ref accessor_tests.hpp.
       Possible interfaces:
       - runtime alias
\verbatim
alias<arg_t, dimension<3> > field1(-3); //records the offset -3 as dynamic value
\endverbatim
       field1(args...) is then equivalent to arg_t(dimension<3>(-3), args...)
       - compiletime alias
\verbatim
        using field1 = alias<arg_t, dimension<7> >::set<-3>;
\endverbatim
       field1(args...) is then equivalent to arg_t(dimension<7>(-3), args...)

       NOTE: noone checks that you did not specify the same dimension twice. If that happens, the first occurrence of
the dimension is chosen
    */
    template < typename AccessorType, typename... Known >
    struct alias {
        GRIDTOOLS_STATIC_ASSERT(is_accessor< AccessorType >::value,
            "wrong type. If you want to generalize the alias "
            "to something more generic than an offset_tuple "
            "remove this assert.");
        GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_dimension< Known >::value...), "wrong type");

        template < int_t Arg1, int_t Arg2 >
        struct pair_ {
            static const constexpr int_t first = Arg1;
            static const constexpr int_t second = Arg2;
        };

        /**
           @brief compile-time aliases, the offsets specified in this way are assured to be compile-time

           This type alias allows to embed some of the offsets directly inside the type of the accessor placeholder.
           For a usage example check the examples folder
        */
        template < int_t... Args >
        using set = accessor_mixed< AccessorType, pair_< Known::direction, Args >... >;

        /**@brief constructor
       \param args are the offsets which are already known*/
        template < typename... Args >
        GT_FUNCTION constexpr alias(Args /*&&*/... args)
            : m_knowns{(int_t)args...} {}

        typedef boost::mpl::vector< Known... > dim_vector;

        /** @brief operator calls the constructor of the offset_tuple

            \param unknowns are the parameters which were not known beforehand. They might be instances of
            the dimension class. Together with the m_knowns offsets form the arguments to be
            passed to the AccessorType functor (which is normally an instance of offset_tuple)
        */
        template < typename... Unknowns >
        GT_FUNCTION AccessorType /*&&*/ operator()(Unknowns /*&&*/... unknowns) const {
#ifdef PEDANTIC // the runtime arguments are not necessarily dimension<>()
            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_dimension< Unknowns >::value...), "wrong type");
#endif
            return AccessorType(
                dimension< Known::direction >(m_knowns[boost::mpl::find< dim_vector, Known >::type::pos::value])...,
                unknowns...);
        }

      private:
        // store the list of offsets which are already known on an array
        int_t m_knowns[sizeof...(Known)];
    };
#endif // CUDA8 (i.e. CXX11_ENABLED for cpu)
} // namespace gridtools
