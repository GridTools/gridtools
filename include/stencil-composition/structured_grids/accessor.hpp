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
        typedef typename super::offset_tuple_t offset_tuple_t;

#ifdef CXX11_ENABLED

        GT_FUNCTION
        constexpr accessor() : super() {}

#ifndef __CUDACC__
        /**inheriting all constructors from offset_tuple*/
        using super::accessor_base;
#else
        /**@brief constructor forwarding all the arguments
        */
        template < typename... ForwardedArgs >
        GT_FUNCTION constexpr accessor(ForwardedArgs... x)
            : super(x...) {}

        // move ctor
        GT_FUNCTION
        constexpr explicit accessor(accessor< ID, Intend, Extent, Number > &&other) : super(std::move(other)) {}

        // copy ctor
        GT_FUNCTION
        constexpr accessor(accessor< ID, Intend, Extent, Number > const &other) : super(other) {}
#endif
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

#ifdef CUDA8 // (i.e. CXX11_ENABLED for cpu)
    /** Trick to make nvcc understand that the accessor is a constant expression*/
    template < typename Pair >
    constexpr dimension< Pair::first > get_dim() {
        return dimension< Pair::first >{Pair::second};
    }

    /**@brief same as accessor but mixing run-time offsets with compile-time ones

       When we know beforehand that the dimension which we are querying is
       a compile-time one, we can use the static method get_constexpr() to get the offset.
       Otherwise the method get() checks before among the static dimensions, and if the
       queried dimension is not found it looks up in the dynamic dimensions. Note that this
       lookup is anyway done at compile time, i.e. the get() method returns in constant time.
     */
    template < typename ArgType, typename... Pair >
    struct offset_tuple_mixed {

        GRIDTOOLS_STATIC_ASSERT(is_offset_tuple<ArgType>::value, "wrong type");
        typedef offset_tuple_mixed< ArgType, Pair... > type;
        static const ushort_t n_dim = ArgType::n_dim;

        typedef ArgType offset_tuple_t;

        // private:
        // static const constexpr dimension< Pair1::first> p1_{Pair1::second};
        // static const constexpr dimension< Pair2::first > p2_{Pair2::second};
        static const constexpr offset_tuple_t s_args_constexpr{get_dim< Pair >()...};

        offset_tuple_t m_args_runtime;

        typedef boost::mpl::vector< static_int< n_dim - Pair::first >... > coordinates;

      public:

        GT_FUNCTION constexpr offset_tuple_mixed()
            : m_args_runtime() {}

        template < typename... ArgsRuntime,
            typename T = typename boost::enable_if_c< accumulate(logical_and(),
                boost::mpl::or_< boost::is_integral< ArgsRuntime >, is_dimension< ArgsRuntime > >::type::value...) >::
                type >
        GT_FUNCTION constexpr offset_tuple_mixed(ArgsRuntime const &... args)
            : m_args_runtime(args...) {}

        template < typename OffsetTuple, typename T = typename boost::enable_if_c<is_offset_tuple<OffsetTuple>::value>::type >
        GT_FUNCTION constexpr offset_tuple_mixed(OffsetTuple const & arg_)
            : m_args_runtime(arg_) {
        }

        template < typename OtherAcc >
        GT_FUNCTION constexpr offset_tuple_mixed(offset_tuple_mixed< OtherAcc, Pair... > &&other_)
            : m_args_runtime(other_.m_args_runtime) {}

        template < typename OtherAcc >
        GT_FUNCTION constexpr offset_tuple_mixed(offset_tuple_mixed< OtherAcc, Pair... > const &other_)
            : m_args_runtime(other_.m_args_runtime) {}

        /**@brief returns the offset at a specific index Idx

           the lookup for the index Idx is done at compile time, i.e. this method returns in constant time
         */
        template < short_t Idx >
        GT_FUNCTION static constexpr int_t get_constexpr() {
#ifndef __CUDACC__
            GRIDTOOLS_STATIC_ASSERT(Idx < s_args_constexpr.n_dim, "the idx must be smaller than the arg dimension");
            GRIDTOOLS_STATIC_ASSERT(Idx >= 0, "the idx must be larger than 0");
            GRIDTOOLS_STATIC_ASSERT(s_args_constexpr.template get< Idx >() >= 0,
                "there is a negative offset. If you did this on purpose recompile with the PEDANTIC_DISABLED flag on.");
#endif
            return s_args_constexpr.template get< Idx >();
        }

        /**@brief returns the offset at a specific index Idx

           the lookup for the index Idx is done at compile time, i.e. this method returns in constant time
         */
        template < short_t Idx >
        GT_FUNCTION constexpr int_t get() const {
            return boost::is_same< typename boost::mpl::find< coordinates, static_int< Idx > >::type,
                       typename boost::mpl::end< coordinates >::type >::type::value
                       ? m_args_runtime.template get< Idx >()
                       : s_args_constexpr.template get< Idx >();
        }
    };

    template <typename ... T>
    struct is_offset_tuple<offset_tuple_mixed<T ...> > : boost::mpl::true_ {};

    template < typename ArgType, typename... Pair >
    constexpr typename offset_tuple_mixed< ArgType, Pair... >::offset_tuple_t offset_tuple_mixed< ArgType, Pair... >::s_args_constexpr;

    template < typename ArgType, typename... Pair >
    struct accessor_mixed : public offset_tuple_mixed<typename ArgType::offset_tuple_t, Pair ...> {
        typedef typename ArgType::index_type index_type;
        typedef typename ArgType::base_t base_t;

        using super = offset_tuple_mixed<typename ArgType::offset_tuple_t, Pair ...>;

        /**inheriting all constructors from offset_tuple*/
        using super::offset_tuple_mixed;

#ifdef __CUDACC__
        template <typename ... T>
        GT_FUNCTION
        constexpr accessor_mixed(T const& ... t_):super(t_ ...){}
#endif

        GT_FUNCTION
        constexpr const super& offsets() const { return *this; }

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
#endif // CUDA8 (i.e. CXX11_ENABLED for cpu)

#ifdef CXX11_ENABLED
    template < uint_t ID, typename Extent = extent< 0, 0, 0, 0, 0, 0 >, ushort_t Number = 3 >
    using in_accessor = accessor< ID, enumtype::in, Extent, Number >;

    template < uint_t ID, typename Extent = extent< 0, 0, 0, 0, 0, 0 >, ushort_t Number = 3 >
    using inout_accessor = accessor< ID, enumtype::inout, Extent, Number >;
#endif

} // namespace gridtools
