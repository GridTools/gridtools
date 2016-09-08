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
#include "../accessor_base.hpp"
#include "../arg.hpp"
#include "../dimension.hpp"
/**
   @file

   @brief File containing the definition of the placeholders used to
   address the storage from whithin the functors.  A placeholder is an
   implementation of the proxy design pattern for the storage class,
   i.e. it is a light object used in place of the storage when
   defining the high level computations, and it will be bound later on
   with a specific instantiation of a storage class.

   Two different types of placeholders are considered:

   - arg represents the storage in the body of the main function, and
     it gets lazily assigned to a real storage.

   - accessor represents the storage inside the functor struct
     containing a Do method. It can be instantiated directly in the Do
     method, or it might be a constant expression instantiated outside
     the functor scope and with static duration.
*/

namespace gridtools {

    /**
       @brief the definition of accessor visible to the user

       \tparam ID the integer unic ID of the field placeholder

       \tparam Extent the extent of i/j indices spanned by the
               placeholder, in the form of <i_minus, i_plus, j_minus,
               j_plus>.  The values are relative to the current
               position. See e.g. horizontal_diffusion::out_function
               as a usage example.

       \tparam Number the number of dimensions accessed by the
               field. Notice that we don't distinguish at this level what we
               call "space dimensions" from the "field dimensions". Both of
               them are accessed using the same interface. whether they are
               field dimensions or space dimension will be decided at the
               moment of the storage instantiation (in the main function)
     */
    template < uint_t ID,
        enumtype::intend Intend = enumtype::in,
        typename Extent = extent< 0, 0, 0, 0, 0, 0 >,
        ushort_t Number = 3 >
    struct accessor : public accessor_base< ID, Intend, Extent, Number > {
        typedef accessor_base< ID, Intend, Extent, Number > super;
        typedef typename super::index_type index_type;
#ifdef CXX11_ENABLED

        GT_FUNCTION
        constexpr accessor() : super() {}

        /**inheriting all constructors from offset_tuple*/
        using super::accessor_base;

#else

        // copy ctor
        GT_FUNCTION
        constexpr explicit accessor(accessor< ID, Intend, Extent, Number > const &other) : super(other) {}

        // copy ctor from an accessor with different ID
        template < ushort_t OtherID >
        GT_FUNCTION constexpr explicit accessor(const accessor< OtherID, Intend, Extent, Number > &other)
            : super(static_cast< accessor_base< OtherID, Intend, Extent, Number > >(other)) {}

        GT_FUNCTION
        constexpr explicit accessor() : super() {}

        /** @brief constructor forwarding all the arguments*/
        template < typename X, typename Y, typename Z, typename T >
        GT_FUNCTION constexpr accessor(X x, Y y, Z z, T t)
            : super(x, y, z, t) {}

        /** @brief constructor forwarding all the arguments*/
        template < typename X, typename Y, typename Z >
        GT_FUNCTION constexpr accessor(X x, Y y, Z z)
            : super(x, y, z) {}

        /** @brief constructor forwarding all the arguments*/
        template < typename X >
        GT_FUNCTION constexpr accessor(X x)
            : super(x) {}

        /** @brief constructor forwarding all the arguments*/
        template < typename X, typename Y >
        GT_FUNCTION constexpr accessor(X x, Y y)
            : super(x, y) {}

#endif
    };

#if defined(CXX11_ENABLED) && !defined(CUDA_CXX11_BUG_1) && !defined(__INTEL_COMPILER)

    /**@brief same as accessor but mixing run-time offsets with compile-time ones

       When we know beforehand that the dimension which we are querying is
       a compile-time one, we can use the static method get_constexpr() to get the offset.
       Otherwise the method get() checks before among the static dimensions, and if the
       queried dimension is not found it looks up in the dynamic dimensions. Note that this
       lookup is anyway done at compile time, i.e. the get() method returns in constant time.
     */
    template < typename ArgType, typename... Pair >
    struct accessor_mixed {

        typedef accessor_mixed< ArgType, Pair... > type;
        static const ushort_t n_dim = ArgType::n_dim;
        typedef typename ArgType::base_t base_t;
        typedef typename ArgType::index_type index_type;

      private:
        // vector of dimensions defined at compiled time for the offsets
        typedef boost::mpl::vector< static_int< n_dim - Pair::first >... > coordinates;

        typedef offset_tuple_mixed< coordinates, ArgType::n_dim, Pair... > offset_tuple_mixed_t;

        typedef offset_tuple< ArgType::n_dim, ArgType::n_dim > offset_tuple_t;

        const offset_tuple_mixed_t m_offsets;

        // compile time offset tuple
        static constexpr offset_tuple_t s_static_offset_tuple{dimension< Pair::first >{Pair::second}...};

      public:
        template < typename... ArgsRuntime >
        GT_FUNCTION constexpr accessor_mixed(const ArgsRuntime... args)
            : m_offsets(args...) {}

        /**@brief returns the offset at a specific index Idx

           the lookup for the index Idx is done at compile time, i.e. this method returns in constant time
         */

        template < short_t Idx >
        GT_FUNCTION constexpr const int_t get() const {
            return m_offsets.template get< Idx >();
        }

        template < short_t Idx >
        GT_FUNCTION static constexpr uint_t const get_constexpr() {
            GRIDTOOLS_STATIC_ASSERT(
                Idx < s_static_offset_tuple.n_dim, "the idx must be smaller than the arg dimension");
            GRIDTOOLS_STATIC_ASSERT(Idx >= 0, "the idx must be larger than 0");
            GRIDTOOLS_STATIC_ASSERT(s_static_offset_tuple.template get< Idx >() >= 0,
                "there is a negative offset. If you did this on purpose recompile with the PEDANTIC_DISABLED flag on.");
            return s_static_offset_tuple.template get< Idx >();
        }

        GT_FUNCTION constexpr offset_tuple_mixed_t const &offsets() const { return m_offsets; }
    };

    template < typename ArgType, typename... Pair >
    constexpr const offset_tuple< ArgType::n_dim, ArgType::n_dim >
        accessor_mixed< ArgType, Pair... >::s_static_offset_tuple;

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

        template < int Arg1, int Arg2 >
        struct pair_ {
            static const constexpr int first = Arg1;
            static const constexpr int second = Arg2;
        };

        /**
           @brief compile-time aliases, the offsets specified in this way are assured to be compile-time

           This type alias allows to embed some of the offsets directly inside the type of the accessor placeholder.
           For a usage example check the examples folder
        */
        template < int... Args >
        using set = accessor_mixed< AccessorType, pair_< Known::direction, Args >... >;

        /**@brief constructor
       \param args are the offsets which are already known*/
        template < typename... Args >
        GT_FUNCTION constexpr alias(Args /*&&*/... args)
            : m_knowns{args...} {}

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
#endif

#ifdef CXX11_ENABLED
    template < uint_t ID, typename Extent = extent< 0, 0, 0, 0, 0, 0 >, ushort_t Number = 3 >
    using in_accessor = accessor< ID, enumtype::in, Extent, Number >;

    template < uint_t ID, typename Extent = extent< 0, 0, 0, 0, 0, 0 >, ushort_t Number = 3 >
    using inout_accessor = accessor< ID, enumtype::inout, Extent, Number >;
#endif

} // namespace gridtools
