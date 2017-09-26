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

#include <boost/mpl/contains.hpp>

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/accumulate_tparams_until.hpp"
#include "../common/generic_metafunctions/copy_into_set.hpp"
#include "../common/generic_metafunctions/is_predicate.hpp"
#include "../common/generic_metafunctions/binary_ops.hpp"
#include "esf.hpp"
#include "independent_esf.hpp"

#ifdef STRUCTURED_GRIDS
#include "structured_grids//accessor_metafunctions.hpp"
#include "structured_grids/esf_metafunctions.hpp"
#else
#include "icosahedral_grids/accessor_metafunctions.hpp"
#include "icosahedral_grids/esf_metafunctions.hpp"
#endif

namespace gridtools {

    /** Metafunction checking if an ESF has, as argument, a given placeholder
    */
    template < typename Arg >
    struct esf_has_parameter_h {
        template < typename Esf >
        struct apply {
            typedef typename boost::mpl::contains< typename Esf::args_t, Arg >::type type;
        };
    };

    /**
       Given an ESF this metafunction provides the list of placeholders (if Pred derives
       from false_type), or map between placeholders in this ESF and the extents
       associated with it (if Pred derives from true_type)
     */
    template < typename Esf, typename Pred = boost::false_type >
    struct esf_args {
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< Esf >::value), "Wrong Type");
        GRIDTOOLS_STATIC_ASSERT((is_meta_predicate< Pred >::type::value), "Not a Predicate");

        typedef typename boost::mpl::if_< Pred, typename Esf::args_with_extents, typename Esf::args_t >::type type;
    };

    /**
       Given an ESF this metafunction provides the placeholder (if Pred derives
       from false_type) at a given index in the list of placeholders, or mpl::pair of
       placeholder and extent (if Pred derives from true_type)
     */
    template < typename Esf, typename Pred >
    struct esf_get_arg_at {
        template < typename Index >
        struct apply {
            GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< Esf >::value), "Wrong Type");
            GRIDTOOLS_STATIC_ASSERT((is_meta_predicate< Pred >::type::value), "Not a Predicate");
            typedef typename boost::mpl::at< typename Esf::args_t, Index >::type placeholder_type;
            typedef typename boost::mpl::if_<
                Pred,
                typename boost::mpl::pair< placeholder_type,
                    typename boost::mpl::at< typename Esf::args_with_extents, placeholder_type >::type >::type,
                typename boost::mpl::at< typename Esf::args_t, Index >::type >::type type;
        };
    };

    /** Provide true_type if the placeholder, which index is Index in the list of placeholders of
        Esf, corresponds to a temporary that is written.
     */
    template < typename Esf >
    struct is_written_temp {
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< Esf >::value), "Wrong Type");
        template < typename Index >
        struct apply {
            typedef typename esf_arg_list< Esf >::type arg_list_t;
            typedef typename boost::mpl::if_<
                is_tmp_arg< typename boost::mpl::at< typename Esf::args_t, Index >::type >,
                typename boost::mpl::if_< is_accessor_readonly< typename boost::mpl::at< arg_list_t, Index >::type >,
                    boost::false_type,
                    boost::true_type >::type,
                boost::false_type >::type type;
        };
    };

    /** Provide true_type if the placeholder, which index is Index in the list of placeholders of
        Esf, correspond to a field (temporary or not) that is is written.
     */
    template < typename Esf >
    struct is_written {
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< Esf >::value), "Wrong Type");
        template < typename Index >
        struct apply {
            typedef typename boost::mpl::if_< is_arg< typename boost::mpl::at< typename Esf::args_t, Index >::type >,
                typename boost::mpl::if_< typename is_accessor_readonly< typename boost::mpl::
                                                  at< typename esf_arg_list< Esf >::type, Index >::type >::type,
                                                  boost::false_type,
                                                  boost::true_type >::type,
                boost::false_type >::type type;
        };
    };

    /**
        If Pred derives from false_type, ::type provide a mpl::vector of placeholders
        that corresponds to temporary fields that are written by EsfF.

        If Pred derives from true_type, ::type provide a mpl::vector of pairs of
        placeholders and extents that corresponds to temporary fields that are written by EsfF.
     */
    template < typename EsfF, typename Pred = boost::false_type >
    struct esf_get_w_temps_per_functor {
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< EsfF >::value), "Wrong Type");
        GRIDTOOLS_STATIC_ASSERT((is_meta_predicate< Pred >::type::value), "Not a Predicate");
        typedef boost::mpl::range_c< uint_t, 0, boost::mpl::size< typename EsfF::args_t >::type::value > iter_range;
        typedef typename boost::mpl::fold<
            iter_range,
            boost::mpl::vector0<>,
            boost::mpl::if_< typename is_written_temp< EsfF >::template apply< boost::mpl::_2 >,
                boost::mpl::push_back< boost::mpl::_1,
                                 typename esf_get_arg_at< EsfF, Pred >::template apply< boost::mpl::_2 > >,
                boost::mpl::_1 > >::type type;
    };

    /**
        If Pred derives from false_type, ::type provide a mpl::vector of placeholders
        that corresponds to fields that are read by EsfF.

        If Pred derives from true_type, ::type provide a mpl::vector of pairs of
        placeholders and extents that corresponds to fields that are read by EsfF.
     */
    template < typename EsfF, typename Pred = boost::false_type >
    struct esf_get_r_temps_per_functor {
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< EsfF >::value), "Wrong Type");
        GRIDTOOLS_STATIC_ASSERT((is_meta_predicate< Pred >::type::value), "Not a Predicate");
        typedef boost::mpl::range_c< uint_t, 0, boost::mpl::size< typename EsfF::args_t >::type::value > range;
        typedef typename boost::mpl::fold<
            range,
            boost::mpl::vector0<>,
            boost::mpl::if_< typename is_written_temp< EsfF >::template apply< boost::mpl::_2 >,
                boost::mpl::_1,
                boost::mpl::push_back< boost::mpl::_1,
                                 typename esf_get_arg_at< EsfF, Pred >::template apply< boost::mpl::_2 > > > >::type
            type;
    };

    /**
        If Pred derives from false_type, ::type provide a mpl::vector of placeholders
        that corresponds to fields (temporary or not) that are written by EsfF.

        If Pred derives from true_type, ::type provide a mpl::vector of pairs of
        placeholders and extents that corresponds to fields (temporary or not) that are
        written by EsfF.
     */
    template < typename EsfF, typename Pred = boost::false_type >
    struct esf_get_w_per_functor {
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< EsfF >::value), "Wrong Type");
        GRIDTOOLS_STATIC_ASSERT((is_meta_predicate< Pred >::type::value), "Not a Predicate");
        typedef boost::mpl::range_c< uint_t, 0, boost::mpl::size< typename EsfF::args_t >::type::value > range;
        typedef typename boost::mpl::fold<
            range,
            boost::mpl::vector0<>,
            boost::mpl::if_< typename is_written< EsfF >::template apply< boost::mpl::_2 >,
                boost::mpl::push_back< boost::mpl::_1,
                                 typename esf_get_arg_at< EsfF, Pred >::template apply< boost::mpl::_2 > >,
                boost::mpl::_1 > >::type type;
    };

    /**
       If the ESF stencil operator writes only one parameter (temporary or
       not) corresponding to a placeholder, it returns this placeholder,
       otherwise it returns the first placeholder to a field that is
       written (temporary or not).
     */
    template < typename EsfF, typename Pred = boost::false_type >
    struct esf_get_the_only_w_per_functor {
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< EsfF >::value), "Wrong Type");
        GRIDTOOLS_STATIC_ASSERT((is_meta_predicate< Pred >::type::value), "Not a Predicate");
        GRIDTOOLS_STATIC_ASSERT(
            (boost::mpl::size< typename esf_get_w_per_functor< EsfF, Pred >::type >::type::value == 1),
            "Each ESF should have a single output argument");
        typedef typename boost::mpl::at_c< typename esf_get_w_per_functor< EsfF >::type, 0 >::type type;
    };

    /**
        If Pred derives from false_type, ::type provide a mpl::vector of placeholders
        that corresponds to fields (temporary or not) that are read by EsfF.

        If Pred derives from true_type, ::type provide a mpl::vector of pairs of
        placeholders and extents that corresponds to fields (temporary or not) that are
        read by EsfF.
     */
    template < typename EsfF, typename Pred = boost::false_type >
    struct esf_get_r_per_functor {
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< EsfF >::value), "Wrong Type");
        GRIDTOOLS_STATIC_ASSERT((is_meta_predicate< Pred >::type::value), "Not a Predicate");
        typedef boost::mpl::range_c< uint_t, 0, boost::mpl::size< typename EsfF::args_t >::type::value > range;
        typedef typename boost::mpl::fold<
            range,
            boost::mpl::vector0<>,
            boost::mpl::if_< typename is_written< EsfF >::template apply< boost::mpl::_2 >,
                boost::mpl::_1,
                boost::mpl::push_back< boost::mpl::_1,
                                 typename esf_get_arg_at< EsfF, Pred >::template apply< boost::mpl::_2 > > > >::type
            type;
    };

    /**
       @brief It computes an associative sequence of all arg types specified by the user
        that are written into by at least one ESF
     */
    template < typename EsfSequence >
    struct compute_readwrite_args {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< EsfSequence, is_esf_descriptor >::value), "Wrong Type");
        typedef typename boost::mpl::fold< EsfSequence,
            boost::mpl::set0<>,
            copy_into_set< esf_get_w_per_functor< boost::mpl::_2 >, boost::mpl::_1 > >::type type;
    };

    /**
       @brief It computes an associative sequence of all arg types specified by the user
        that are readonly through all ESFs/MSSs
     */
    template < typename EsfSequence >
    struct compute_readonly_args {
        template < typename Acc, typename Esf, typename ReadWriteArgs >
        struct extract_readonly_arg {
            typedef typename boost::mpl::fold< typename Esf::args_t,
                Acc,
                boost::mpl::if_< boost::mpl::or_< boost::mpl::has_key< ReadWriteArgs, boost::mpl::_2 >,
                                     is_tmp_arg< boost::mpl::_2 > >,
                                                   boost::mpl::_1,
                                                   boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > > >::type type;
        };

        // compute all the args which are written by at least one ESF
        typedef typename compute_readwrite_args< EsfSequence >::type readwrite_args_t;

        typedef typename boost::mpl::fold< EsfSequence,
            boost::mpl::set0<>,
            extract_readonly_arg< boost::mpl::_1, boost::mpl::_2, readwrite_args_t > >::type type;
    };

    /**
       @brief It computes an associative sequence of indices for all arg types specified by
        the user that are readonly through all ESFs/MSSs
     */
    template < typename EsfSequence >
    struct compute_readonly_args_indices {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< EsfSequence, is_esf_descriptor >::value), "Wrong type");
        typedef typename boost::mpl::fold< typename compute_readonly_args< EsfSequence >::type,
            boost::mpl::set0<>,
            boost::mpl::insert< boost::mpl::_1, arg_index< boost::mpl::_2 > > >::type type;
    };

#ifdef CUDA8
    /*
      Given an array of pairs (placeholder, extent) checks if all
      extents are the same and equal to the extent passed in
     */
    template < typename VectorOfPairs, typename Extent, ushort_t Limit >
    struct check_all_extents_are_same_upto {
        template < typename Pair >
        struct _check {
            using type = static_bool<
                accumulate_tparams_until< int_t, equal, logical_and, typename Pair::second, Extent, Limit >::value >;
        };

        typedef typename is_sequence_of< VectorOfPairs, _check >::type type;
    };

#endif

    /*
      Given an array of pairs (placeholder, extent) checks if all
      extents are the same and equal to the extent passed in
     */
    template < typename VectorOfPairs, typename Extent >
    struct check_all_extents_are {
        template < typename Pair >
        struct _check {
            typedef typename boost::is_same< typename Pair::second, Extent >::type type;
        };

        typedef typename is_sequence_of< VectorOfPairs, _check >::type type;
    };

    template < typename T >
    struct is_esf_descriptor< independent_esf< T > > : boost::mpl::true_ {};

    // Takes a list of esfs and independent_esf and produces a list of esfs, with the independent unwrapped
    template < typename ESFList >
    struct unwrap_independent {

        GRIDTOOLS_STATIC_ASSERT(
            (is_sequence_of< ESFList, is_esf_descriptor >::value), "Error: ESFList must be a list of ESFs");

        template < typename CurrentList, typename CurrentElement >
        struct populate {
            typedef typename boost::mpl::push_back< CurrentList, CurrentElement >::type type;
        };

        template < typename CurrentList, typename IndependentList >
        struct populate< CurrentList, independent_esf< IndependentList > > {
            typedef typename boost::mpl::fold< IndependentList,
                CurrentList,
                populate< boost::mpl::_1, boost::mpl::_2 > >::type type;
        };

        typedef typename boost::mpl::fold< ESFList,
            boost::mpl::vector0<>,
            populate< boost::mpl::_1, boost::mpl::_2 > >::type type;
    }; // struct unwrap_independent

    /** Retrieve the extent in esf_descriptor_with_extents

       \tparam Esf The esf_descriptor that must be the one speficying the extent
    */
    template < typename Esf >
    struct esf_extent;

    template < typename ESF, typename Extent, typename ArgArray, typename Staggering >
    struct esf_extent< esf_descriptor_with_extent< ESF, Extent, ArgArray, Staggering > > {
        using type = Extent;
    };

} // namespace gridtools
