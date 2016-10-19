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
        typedef typename ArgType::index_type index_type;
        typedef typename ArgType::base_t base_t;
        typedef typename ArgType::offset_tuple_t offset_tuple_t;
        typedef typename ArgType::extent_t extent_t;

        using super = offset_tuple_mixed< typename ArgType::offset_tuple_t, Pair... >;
        /**inheriting all constructors from offset_tuple*/
        using typename super::offset_tuple_mixed;

#if defined(__CUDACC__) || defined(__clang__)
        // the protection for the arguments is done in offset_tuple constructors
        template < typename... T >
        GT_FUNCTION constexpr accessor_mixed(T const &... t_)
            : super(t_...) {}
#endif

        GT_FUNCTION
        constexpr const super &offsets() const { return *this; }
    };

    template < uint_t ID,
        enumtype::intend Intend = enumtype::in,
        typename Extent = extent< 0, 0, 0, 0, 0, 0 >,
        ushort_t Number = 3 >
    struct accessor : accessor_mixed< accessor_impl< ID, Intend, Extent, Number > > {
        using accessor_mixed< accessor_impl< ID, Intend, Extent, Number > >::accessor_mixed;
    };

    template < uint_t ID, typename Extent = extent< 0, 0, 0, 0, 0, 0 >, ushort_t Number = 3 >
    using in_accessor = accessor< ID, enumtype::in, Extent, Number >;

    template < uint_t ID, typename Extent = extent< 0, 0, 0, 0, 0, 0 >, ushort_t Number = 3 >
    using inout_accessor = accessor< ID, enumtype::inout, Extent, Number >;

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
