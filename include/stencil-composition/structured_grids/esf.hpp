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

#include <boost/type_traits/is_const.hpp>

#include "../../common/generic_metafunctions/is_sequence_of.hpp"
#include "../aggregator_type.hpp"
#include "../esf_aux.hpp"
#include "../esf_fwd.hpp"
#include "../expandable_parameters/vector_accessor.hpp"
#include "../sfinae.hpp"
#include "accessor.hpp"
#include "accessor_mixed.hpp"

/**
   @file
   @brief Descriptors for Elementary Stencil Function (ESF)
*/
namespace gridtools {

    namespace _impl {
        /**
           Metafunction to check that the arg_list mpl::vector list the
           different accessors in order!
        */
        template < typename ArgList >
        struct check_arg_list {
            template < typename Reduced, typename Element >
            struct _check {
                typedef typename boost::mpl::if_c< (Element::index_t::value == Reduced::value + 1),
                    boost::mpl::int_< Reduced::value + 1 >,
                    boost::mpl::int_< -Reduced::value - 1 > >::type type;
            };

            typedef typename boost::mpl::fold< ArgList,
                boost::mpl::int_< -1 >,
                _check< boost::mpl::_1, boost::mpl::_2 > >::type res_type;

            typedef typename boost::mpl::if_c< (res_type::value + 1 == boost::mpl::size< ArgList >::value),
                boost::true_type,
                boost::false_type >::type type;

            static const bool value = type::value;
        };

        /**
           \brief returns the index chosen when the placeholder U was defined
        */
        struct l_get_index {
            template < typename U >
            struct apply {
                typedef static_uint< U::index_t::value > type;
            };
        };

        template < typename OriginalPlaceholders >
        struct compute_index_set {

            /**
             * \brief Get a sequence of the same type as original_placeholders, containing the indexes relative to the
             * placehoolders
             * note that the static const indexes are transformed into types using mpl::integral_c
             */
            typedef typename boost::mpl::transform< OriginalPlaceholders, l_get_index >::type raw_index_list;

            /**@brief length of the index list eventually with duplicated indices */
            static const uint_t len = boost::mpl::size< raw_index_list >::value;

            /**
               @brief filter out duplicates
               check if the indexes are repeated (a common error is to define 2 types with the same index)
            */
            typedef typename boost::mpl::fold< raw_index_list,
                boost::mpl::set<>,
                boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > >::type index_set;
        };

    } // namespace _impl

    /**
     * @brief Descriptors for Elementary Stencil Function (ESF)
     */
    template < typename ESF, typename ArgArray, typename Staggering = staggered< 0, 0, 0, 0, 0, 0 > >
    struct esf_descriptor {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< ArgArray, is_arg >::value),
            "wrong types for the list of parameter placeholders\n"
            "check the make_stage syntax");

      public:
        typedef ESF esf_function;
        typedef ArgArray args_t;

        /** Type member with the mapping between placeholder types (as key) to extents in the operator */
        typedef
            typename impl::make_arg_with_extent_map< args_t, typename esf_function::arg_list >::type args_with_extents;
        typedef Staggering staggering_t;

        //////////////////////Compile time checks ////////////////////////////////////////////////////////////
        BOOST_MPL_HAS_XXX_TRAIT_DEF(arg_list)
        GRIDTOOLS_STATIC_ASSERT(has_arg_list< esf_function >::type::value,
            "The type arg_list was not found in a user functor definition. All user functors must have a type alias "
            "called \'arg_list\', which is an MPL vector containing the list of accessors defined in the functor "
            "(NOTE: the \'global_accessor\' types are excluded from this list). Example: \n\n using v1=accessor<0>; \n "
            "using v2=global_accessor<1, enumtype::in>; \n using v3=accessor<2>; \n using "
            "arg_list=boost::mpl::vector<v1, v3>;");

        GRIDTOOLS_STATIC_ASSERT(_impl::check_arg_list< typename esf_function::arg_list >::value,
            "The list of accessors in a user functor (i.e. the arg_list type to be defined on each functor) does not "
            "have increasing index");

        /**
         * \brief Get a sequence of the same type as original_placeholders, containing the indexes relative to the
         * placehoolders
         * note that the static const indexes are transformed into types using mpl::integral_c
         */
        typedef _impl::compute_index_set< typename esf_function::arg_list > check_holes;
        typedef typename check_holes::raw_index_list raw_index_list;
        typedef typename check_holes::index_set index_set;
        static const ushort_t len = check_holes::len;

        // actual check if the user specified placeholder arguments with the same index
        GRIDTOOLS_STATIC_ASSERT((len == boost::mpl::size< index_set >::type::value),
            "You specified different accessors with the same index. Check the indexes of the accessor definitions.");

        // checking if the index list contains holes (a common error is to define a list of types with indexes which are
        // not contiguous)
        typedef typename boost::mpl::find_if< raw_index_list,
            boost::mpl::greater< boost::mpl::_1, static_int< len - 1 > > >::type test;
        // check if the index list contains holes (a common error is to define a list of types with indexes which are
        // not contiguous)
        GRIDTOOLS_STATIC_ASSERT((boost::is_same< typename test::type, boost::mpl::void_ >::value),
            "the index list contains holes:\n "
            "The numeration of the placeholders is not contiguous. You have to define each accessor with a unique "
            "identifier ranging "
            " from 1 to N without \"holes\".");
        //////////////////////////////////////////////////////////////////////////////////////////////////////
    };

    template < typename T, typename V >
    std::ostream &operator<<(std::ostream &s, esf_descriptor< T, V > const7) {
        return s << "esf_desctiptor< " << T() << " with "
                 << boost::mpl::size< typename esf_descriptor< T, V >::args_t >::type::value
                 << " arguments (double check "
                 << boost::mpl::size< typename esf_descriptor< T, V >::esf_function::arg_list >::type::value << ")";
    }

    template < typename ESF, typename ArgArray, typename Staggering >
    struct is_esf_descriptor< esf_descriptor< ESF, ArgArray, Staggering > > : boost::mpl::true_ {};

    template < typename ESF, typename Extent, typename ArgArray, typename Staggering = staggered< 0, 0, 0, 0, 0, 0 > >
    struct esf_descriptor_with_extent : public esf_descriptor< ESF, ArgArray, Staggering > {
        GRIDTOOLS_STATIC_ASSERT((is_extent< Extent >::value), "stage descriptor is expecting a extent type");
    };

    template < typename ESF, typename Extent, typename ArgArray, typename Staggering >
    struct is_esf_descriptor< esf_descriptor_with_extent< ESF, Extent, ArgArray, Staggering > > : boost::mpl::true_ {};

    template < typename ESF >
    struct is_esf_with_extent : boost::mpl::false_ {
        GRIDTOOLS_STATIC_ASSERT(is_esf_descriptor< ESF >::type::value,
            GT_INTERNAL_ERROR_MSG("is_esf_with_extents expects an esf_descripto as template argument"));
    };

    template < typename ESF, typename Extent, typename ArgArray, typename Staggering >
    struct is_esf_with_extent< esf_descriptor_with_extent< ESF, Extent, ArgArray, Staggering > > : boost::mpl::true_ {};

} // namespace gridtools
