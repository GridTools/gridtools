/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <boost/mpl/set.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/type_traits/is_const.hpp>

#include "../../common/generic_metafunctions/is_sequence_of.hpp"
#include "../esf_aux.hpp"
#include "../esf_fwd.hpp"
#include "extent.hpp"

/**
   @file
   @brief Descriptors for Elementary Stencil Function (ESF)
*/
namespace gridtools {

    namespace _impl {
        /**
           Metafunction to check that the param_list mpl::vector list the
           different accessors in order!
        */
        template <typename ArgList>
        struct check_param_list {
            template <typename Reduced, typename Element>
            struct _check {
                typedef typename boost::mpl::if_c<(Element::index_t::value == Reduced::value + 1),
                    boost::mpl::int_<Reduced::value + 1>,
                    boost::mpl::int_<-Reduced::value - 1>>::type type;
            };

            typedef
                typename boost::mpl::fold<ArgList, boost::mpl::int_<-1>, _check<boost::mpl::_1, boost::mpl::_2>>::type
                    res_type;

            typedef typename boost::mpl::if_c<(res_type::value + 1 == boost::mpl::size<ArgList>::value),
                boost::true_type,
                boost::false_type>::type type;

            static const bool value = type::value;
        };

        /**
           \brief returns the index chosen when the placeholder U was defined
        */
        struct l_get_index {
            template <typename U>
            struct apply {
                typedef static_uint<U::index_t::value> type;
            };
        };

        template <typename OriginalPlaceholders>
        struct compute_index_set {

            /**
             * \brief Get a sequence of the same type as original_placeholders, containing the indexes relative to the
             * placehoolders
             * note that the static const indexes are transformed into types using mpl::integral_c
             */
            typedef typename boost::mpl::transform<OriginalPlaceholders, l_get_index>::type raw_index_list;

            /**@brief length of the index list eventually with duplicated indices */
            static const uint_t len = boost::mpl::size<raw_index_list>::value;

            /**
               @brief filter out duplicates
               check if the indexes are repeated (a common error is to define 2 types with the same index)
            */
            typedef typename boost::mpl::fold<raw_index_list,
                boost::mpl::set<>,
                boost::mpl::insert<boost::mpl::_1, boost::mpl::_2>>::type index_set;
        };

    } // namespace _impl

    /**
     * @brief Descriptors for Elementary Stencil Function (ESF)
     */
    template <typename ESF, typename ArgArray>
    struct esf_descriptor {
        GT_STATIC_ASSERT((is_sequence_of<ArgArray, is_plh>::value),
            "wrong types for the list of parameter placeholders\n"
            "check the make_stage syntax");

      public:
        typedef ESF esf_function;
        typedef ArgArray args_t;

        /** Type member with the mapping between placeholder types (as key) to extents in the operator */
        typedef
            typename impl::make_arg_with_extent_map<args_t, typename esf_function::param_list>::type args_with_extents;

        //////////////////////Compile time checks ////////////////////////////////////////////////////////////
        BOOST_MPL_HAS_XXX_TRAIT_DEF(param_list)
        GT_STATIC_ASSERT(has_param_list<esf_function>::type::value,
            "The type param_list was not found in a user functor definition. All user functors must have a type alias "
            "called \'param_list\', which is an MPL vector containing the list of accessors defined in the functor "
            "(NOTE: the \'global_accessor\' types are excluded from this list). Example: \n\n using v1=accessor<0>; \n "
            "using v2=global_accessor<1>; \n using v3=accessor<2>; \n using "
            "param_list=boost::mpl::vector<v1, v3>;");

        GT_STATIC_ASSERT(_impl::check_param_list<typename esf_function::param_list>::value,
            "The list of accessors in a user functor (i.e. the param_list type to be defined on each functor) does not "
            "have increasing index");

        /**
         * \brief Get a sequence of the same type as original_placeholders, containing the indexes relative to the
         * placehoolders
         * note that the static const indexes are transformed into types using mpl::integral_c
         */
        typedef _impl::compute_index_set<typename esf_function::param_list> check_holes;
        typedef typename check_holes::raw_index_list raw_index_list;
        typedef typename check_holes::index_set index_set;
        static const ushort_t len = check_holes::len;

        // actual check if the user specified placeholder arguments with the same index
        GT_STATIC_ASSERT((len == boost::mpl::size<index_set>::type::value),
            "You specified different accessors with the same index. Check the indexes of the accessor definitions.");

        // checking if the index list contains holes (a common error is to define a list of types with indexes which are
        // not contiguous)
        typedef
            typename boost::mpl::find_if<raw_index_list, boost::mpl::greater<boost::mpl::_1, static_int<len - 1>>>::type
                test;
        // check if the index list contains holes (a common error is to define a list of types with indexes which are
        // not contiguous)
        GT_STATIC_ASSERT((boost::is_same<typename test::type, boost::mpl::void_>::value),
            "the index list contains holes:\n "
            "The numeration of the placeholders is not contiguous. You have to define each accessor with a unique "
            "identifier ranging "
            " from 1 to N without \"holes\".");
        //////////////////////////////////////////////////////////////////////////////////////////////////////
    };

    template <typename ESF, typename ArgArray>
    struct is_esf_descriptor<esf_descriptor<ESF, ArgArray>> : boost::mpl::true_ {};

    template <typename ESF, typename Extent, typename ArgArray>
    struct esf_descriptor_with_extent : esf_descriptor<ESF, ArgArray> {
        GT_STATIC_ASSERT((is_extent<Extent>::value), "stage descriptor is expecting a extent type");
    };

    template <typename ESF, typename Extent, typename ArgArray>
    struct is_esf_descriptor<esf_descriptor_with_extent<ESF, Extent, ArgArray>> : boost::mpl::true_ {};

    template <typename ESF>
    struct is_esf_with_extent : boost::mpl::false_ {
        GT_STATIC_ASSERT(is_esf_descriptor<ESF>::type::value,
            GT_INTERNAL_ERROR_MSG("is_esf_with_extents expects an esf_descripto as template argument"));
    };

    template <typename ESF, typename Extent, typename ArgArray>
    struct is_esf_with_extent<esf_descriptor_with_extent<ESF, Extent, ArgArray>> : boost::mpl::true_ {};

} // namespace gridtools
