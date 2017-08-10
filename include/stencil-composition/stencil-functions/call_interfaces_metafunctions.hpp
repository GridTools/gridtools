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

#include <boost/mpl/count_if.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/type_traits/is_same.hpp>
#include "../../common/defs.hpp"
#include "../structured_grids/accessor_metafunctions.hpp"
#include "./call_interfaces_fwd.hpp"

namespace gridtools {
    namespace _impl {
        template < typename T >
        struct is_function_aggregator : boost::mpl::false_ {};

        template < typename CallerAggregator,
            int Offi,
            int Offj,
            int Offk,
            typename PassedAccessors,
            typename ReturnType,
            int OutArg >
        struct is_function_aggregator<
            function_aggregator< CallerAggregator, Offi, Offj, Offk, PassedAccessors, ReturnType, OutArg > >
            : boost::mpl::true_ {};

        template < typename CallerAggregator,
            int Offi,
            int Offj,
            int Offk,
            typename PassedAccessors,
            typename ReturnType,
            int OutArg >
        struct is_function_aggregator<
            function_aggregator_offsets< CallerAggregator, Offi, Offj, Offk, PassedAccessors, ReturnType, OutArg > >
            : boost::mpl::true_ {};

        template < typename CallerAggregator, int Offi, int Offj, int Offk, typename PassedAccessors >
        struct is_function_aggregator<
            function_aggregator_procedure< CallerAggregator, Offi, Offj, Offk, PassedAccessors > > : boost::mpl::true_ {
        };

        template < typename CallerAggregator, int Offi, int Offj, int Offk, typename PassedAccessors >
        struct is_function_aggregator<
            function_aggregator_procedure_offsets< CallerAggregator, Offi, Offj, Offk, PassedAccessors > >
            : boost::mpl::true_ {};

        /** Metafunction to compute the index of the first accessor in the
            list of accessors to be written.
        */
        template < typename Functor >
        struct _get_index_of_first_non_const {

            typedef typename boost::mpl::find_if< typename Functor::arg_list,
                is_accessor_written< typename boost::mpl::_ > >::type iter;

            typedef typename boost::mpl::if_<
                typename boost::is_same< iter, typename boost::mpl::end< typename Functor::arg_list >::type >::type,
                boost::mpl::int_< -1 >,
                typename iter::pos >::type result;

            static const int value = result::value;
        };

        /** Metafunction to check that there is only one
            written argument in the argument list of a stencil
            operator, so that it is legal to call it as a
            function.

            User protection
        */
        template < typename Functor >
        struct can_be_a_function {
            typedef typename boost::mpl::count_if< typename Functor::arg_list,
                is_accessor_written< boost::mpl::_ > >::type type;

            static const bool value = type::value == 1;
        };

        /** Metafunction to check if a list of types with value traits
            contains a type with a given value.
            This metafunction can search values that are coded in different ways,
            like boost::mpl::int_ and static_int.

            It may belong to common, but its use may be not so easily explained.
         */
        template < typename ListOfIndices, typename Value >
        struct contains_value {
            template < typename TheValue >
            struct has_value {
                template < typename Element >
                struct apply {
                    static const bool value = Value::value == Element::value;
                    using type = boost::mpl::bool_< value >;
                };
            };

            using cnt = typename boost::mpl::count_if< ListOfIndices,
                typename has_value< Value >::template apply< boost::mpl::_ > >::type;
            using type = boost::mpl::bool_< cnt::value >= 1 >;
            static const bool value = type::value;
        };

        /** Metafunction to collect all the indices of a sequence
            corresponding to non-accessors. Used in calling interface
            to provide enable_ifs based on accessor indices of the
            called function.
         */
        template < typename PArguments >
        struct insert_index_if_not_accessor {
            template < typename Index, typename CurrentState >
            struct apply {
                typedef typename boost::mpl::at< PArguments, static_uint< Index::value > >::type to_check;
                typedef typename boost::mpl::if_< is_accessor< to_check >,
                    CurrentState,
                    typename boost::mpl::push_back< CurrentState, Index >::type >::type type;
            };
        };

        /** Struct to wrap a reference that is passed to a procedure
            (call_proc). This is used to deduce the type of the arguments
            required by the called function.
         */
        template < typename Type >
        struct wrap_reference {
            using type = Type;

            type *p_value;

            GT_FUNCTION
            wrap_reference(type const &v) : p_value(const_cast< typename std::decay< type >::type * >(&v)) {}

            GT_FUNCTION
            type &value() const { return *p_value; }
        };

        /** When calling a procedure, a new aggregator has to be
            produced which keeps the accessors as they are and
            wrap the references.
         */
        template < typename... Args >
        struct package_args;

        template < class First, typename... Args >
        struct package_args< First, Args... > {
            using thefirst = typename std::decay< First >::type;
            typedef
                typename boost::mpl::if_c< is_accessor< thefirst >::value, thefirst, wrap_reference< thefirst > >::type
                    to_pack;
            typedef typename boost::mpl::push_front< typename package_args< Args... >::type, to_pack >::type type;
        };

        template < class T >
        struct package_args< T > {
            using thefirst = typename std::decay< T >::type;
            typedef
                typename boost::mpl::if_c< is_accessor< thefirst >::value, thefirst, wrap_reference< thefirst > >::type
                    to_pack;
            typedef boost::mpl::vector1< to_pack > type;
        };

        template <>
        struct package_args<> {
            typedef boost::mpl::vector0<> type;
        };

        /** Maker to wrap an argument if it is not an accessor.

            Used to apply the transformation to a variadic pack.
         */
        template < typename T >
        GT_FUNCTION typename boost::enable_if_c< is_accessor< T >::value, T >::type make_wrap(T const &v) {
            return v;
        }

        template < typename T >
        GT_FUNCTION typename boost::enable_if_c< not is_accessor< T >::value, _impl::wrap_reference< T > >::type
        make_wrap(T const &v) {
            return _impl::wrap_reference< typename std::decay< T >::type >(v);
        }

        /**
        *  Metafunction used to select the right accessor. When using nested 
        *  call interfaces we have to select the accessor with the correct extent.
        *  Case: one of the given types is not an accessor type
        *  @tparam Acc1 Accessor type 1
        *  @tparam Acc2 Accessor type 2
        */
        template <typename Acc1, typename Acc2>
        struct select_accessor {
            typedef Acc1 type;
        };

        /**
        *  Metafunction used to select the right accessor. When using nested 
        *  call interfaces we have to select the accessor with the correct extent.
        *  @tparam I1 Index of accessor 1
        *  @tparam Intend1 intend of accessor 1
        *  @tparam Extend1 extend of accessor 1
        *  @tparam Dim1 dimension of accessor 1
        *  @tparam I2 Index of accessor 2
        *  @tparam Intend2 intend of accessor 2
        *  @tparam Extend2 extend of accessor 2
        *  @tparam Dim2 dimension of accessor 2
        */
        template < uint_t I1, enumtype::intend Intend1, typename Extend1, ushort_t Dim1, 
            uint_t I2, enumtype::intend Intend2, typename Extend2, ushort_t Dim2 >
        struct select_accessor< accessor<I1, Intend1, Extend1, Dim1>, accessor<I2, Intend2, Extend2, Dim2> > {
            // get the enclosing extent
            typedef typename enclosing_extent< Extend1, Extend2 >::type new_extent_t;
            // check which of the accessor matches the enclosing extent
            typedef typename boost::mpl::if_< boost::is_same< Extend1, new_extent_t >, 
                accessor<I2, Intend1, Extend1, Dim1>, 
                accessor<I2, Intend2, Extend2, Dim2> >::type type;
        };

        /**
        *  Metafunction used to extend the extent of a list of accessors.
        *  Case: given type is an mpl vector of accessors.
        *  @tparam Offi offset in I direction
        *  @tparam Offj offset in J direction
        *  @tparam Offk offset in K direction
        *  @tparam Vec mpl vector of accessor types
        */
        template < int_t Offi, int_t Offj, int_t Offk, typename Vec >
        struct extend_accessors {
            typedef Vec acc_t;
            // modify extents
            typedef typename boost::mpl::transform< acc_t, extend_accessor< 0, Offi > >::type mod_acc_i_t;
            typedef typename boost::mpl::transform< mod_acc_i_t, extend_accessor< 1, Offj > >::type mod_acc_j_t;
            typedef typename boost::mpl::transform< mod_acc_j_t, extend_accessor< 2, Offk > >::type type;
        };

        /**
        *  Metafunction used to extend the extent an accessor.
        *  Case: given type is a single accessor.
        *  @tparam Offi offset in I direction
        *  @tparam Offj offset in J direction
        *  @tparam Offk offset in K direction
        *  @tparam I accessor index
        *  @tparam Intend accessor intend
        *  @tparam Extend accessor extend
        *  @tparam Dim accessor dimension
        */
        template < int_t Offi, int_t Offj, int_t Offk, uint_t I, enumtype::intend Intend, typename Extend, ushort_t Dim >
        struct extend_accessors< Offi, Offj, Offk, accessor<I, Intend, Extend, Dim> > {
            // modify extents
            typedef typename extend_accessor< 0, Offi >::template apply< accessor<I, Intend, Extend, Dim> >::type mod_acc_i_t;
            typedef typename extend_accessor< 1, Offj >::template apply<mod_acc_i_t>::type mod_acc_j_t;
            typedef typename extend_accessor< 2, Offk >::template apply<mod_acc_j_t>::type type;
        };        

        /**
        *  Function used to reinstantiate accessors. In case we are 
        *  using call interfaces it is required to reinstantiate the accessor
        *  from e.g., accessor<..., extent<0,0> > to accessor<..., extent<-1,1> >
        *  Case: passed element is accessor type        
        *  @tparam Offi offset in I direction
        *  @tparam Offj offset in J direction
        *  @tparam Offk offset in K direction
        *  @tparam Accessor Input Accessor type
        *  @param a Accessor instance
        *  @return new accessor instance with extended extents
        */
        template < int OffI, int OffJ, int OffK, typename Accessor >
        GT_FUNCTION constexpr typename boost::enable_if_c<is_accessor<Accessor>::value, accessor< Accessor::index_t::value, Accessor::intend_v, typename extend_extent< 0, OffI, typename extend_extent< 1, OffJ, typename extend_extent< 2, OffK, typename Accessor::extent_t >::type >::type >::type,
        Accessor::n_dimensions > >::type extend_accessor_instance(Accessor a) {
            typedef typename boost::enable_if_c<is_accessor<Accessor>::value, accessor< Accessor::index_t::value, Accessor::intend_v, typename extend_extent< 0, OffI, typename extend_extent< 1, OffJ, typename extend_extent< 2, OffK, typename Accessor::extent_t >::type >::type >::type, Accessor::n_dimensions > >::type new_acc_t;
            return new_acc_t(a.offsets());
        }

        /**
        *  Function used to reinstantiate accessors. In case we are 
        *  using call interfaces it is required to reinstantiate the accessor
        *  from e.g., accessor<..., extent<0,0> > to accessor<..., extent<-1,1> >
        *  Case: passed element is no accessor type
        *  @tparam Offi offset in I direction
        *  @tparam Offj offset in J direction
        *  @tparam Offk offset in K direction
        *  @tparam T Input type
        *  @return unmodified value
        */
        template < int OffI, int OffJ, int OffK, typename T >
        GT_FUNCTION constexpr typename boost::enable_if_c<!is_accessor<T>::value, T >::type
        extend_accessor_instance(T a) {
            return a;
        }

    } // namespace _impl
} // namespace gridtools
