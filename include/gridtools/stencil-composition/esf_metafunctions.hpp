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

#include <tuple>

#include <boost/mpl/contains.hpp>

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/binary_ops.hpp"
#include "../common/generic_metafunctions/copy_into_set.hpp"
#include "../common/generic_metafunctions/is_predicate.hpp"
#include "../meta.hpp"
#include "accessor_metafunctions.hpp"
#include "esf.hpp"
#include "independent_esf.hpp"

#ifndef GT_ICOSAHEDRAL_GRIDS
#include "structured_grids/esf_metafunctions.hpp"
#else
#include "icosahedral_grids/esf_metafunctions.hpp"
#endif

namespace gridtools {

    /** Metafunction checking if an ESF has, as argument, a given placeholder
     */
    template <typename Arg>
    struct esf_has_parameter_h {
        template <typename Esf>
        struct apply {
            typedef typename boost::mpl::contains<typename Esf::args_t, Arg>::type type;
        };
    };

    /**
       Given an ESF this metafunction provides the list of placeholders (if Pred derives
       from false_type), or map between placeholders in this ESF and the extents
       associated with it (if Pred derives from true_type)
     */
    template <typename Esf>
    struct esf_args {
        GT_STATIC_ASSERT((is_esf_descriptor<Esf>::value), "Wrong Type");
        typedef typename Esf::args_t type;
    };

    /**
       Given an ESF this metafunction provides the placeholder (if Pred derives
       from false_type) at a given index in the list of placeholders, or mpl::pair of
       placeholder and extent (if Pred derives from true_type)
     */
    template <typename Esf, typename Pred, typename Index>
    struct esf_get_arg_at {
        GT_STATIC_ASSERT((is_esf_descriptor<Esf>::value), "Wrong Type");
        GT_STATIC_ASSERT((is_meta_predicate<Pred>::type::value), "Not a Predicate");
        typedef typename boost::mpl::at<typename Esf::args_t, Index>::type placeholder_type;
        typedef typename boost::mpl::if_<Pred,
            typename boost::mpl::pair<placeholder_type,
                typename boost::mpl::at<typename Esf::args_with_extents, placeholder_type>::type>::type,
            placeholder_type>::type type;
    };

    /** Provide true_type if the placeholder, which index is Index in the list of placeholders of
        Esf, corresponds to a temporary that is written.
     */
    template <typename Esf, typename Index>
    struct is_written_temp {
        GT_STATIC_ASSERT((is_esf_descriptor<Esf>::value), "Wrong Type");
        typedef typename esf_param_list<Esf>::type param_list_t;
        typedef typename boost::mpl::if_<is_tmp_arg<typename boost::mpl::at<typename Esf::args_t, Index>::type>,
            typename boost::mpl::if_<is_accessor_readonly<typename boost::mpl::at<param_list_t, Index>::type>,
                boost::false_type,
                boost::true_type>::type,
            boost::false_type>::type type;
    };

    /** Provide true_type if the placeholder, which index is Index in the list of placeholders of
        Esf, correspond to a field (temporary or not) that is is written.
     */
    template <typename Esf>
    struct is_written {
        GT_STATIC_ASSERT((is_esf_descriptor<Esf>::value), "Wrong Type");
        template <typename Index>
        struct apply {
            typedef typename boost::mpl::if_<is_plh<typename boost::mpl::at<typename Esf::args_t, Index>::type>,
                typename boost::mpl::if_<typename is_accessor_readonly<typename boost::mpl::
                                                 at<typename esf_param_list<Esf>::type, Index>::type>::type,
                    boost::false_type,
                    boost::true_type>::type,
                boost::false_type>::type type;
        };
    };

    template <typename EsfF>
    struct esf_get_w_temps_per_functor {
        GT_STATIC_ASSERT(is_esf_descriptor<EsfF>::value, GT_INTERNAL_ERROR);
        typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename EsfF::args_t>::type::value> iter_range;
        typedef typename boost::mpl::fold<iter_range,
            boost::mpl::vector0<>,
            boost::mpl::if_<is_written_temp<EsfF, boost::mpl::_2>,
                boost::mpl::push_back<boost::mpl::_1, boost::mpl::at<typename EsfF::args_t, boost::mpl::_2>>,
                boost::mpl::_1>>::type type;
    };

    /**
       If Pred derives from false_type, `type` provide a mpl::vector of placeholders
       that corresponds to fields (temporary or not) that are written by EsfF.

       If Pred derives from true_type, `type` provide a mpl::vector of pairs of
       placeholders and extents that corresponds to fields (temporary or not) that are
       written by EsfF.
     */
    template <typename EsfF, typename Pred = boost::false_type>
    struct esf_get_w_per_functor {
        GT_STATIC_ASSERT((is_esf_descriptor<EsfF>::value), "Wrong Type");
        GT_STATIC_ASSERT((is_meta_predicate<Pred>::type::value), "Not a Predicate");
        typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename EsfF::args_t>::type::value> range;
        typedef typename boost::mpl::fold<range,
            boost::mpl::vector0<>,
            boost::mpl::if_<typename is_written<EsfF>::template apply<boost::mpl::_2>,
                boost::mpl::push_back<boost::mpl::_1, esf_get_arg_at<EsfF, Pred, boost::mpl::_2>>,
                boost::mpl::_1>>::type type;
    };

    /**
       If the ESF stencil operator writes only one parameter (temporary or
       not) corresponding to a placeholder, it returns this placeholder,
       otherwise it returns the first placeholder to a field that is
       written (temporary or not).
     */
    template <typename EsfF, typename Pred = boost::false_type>
    struct esf_get_the_only_w_per_functor {
        GT_STATIC_ASSERT((is_esf_descriptor<EsfF>::value), "Wrong Type");
        GT_STATIC_ASSERT((is_meta_predicate<Pred>::type::value), "Not a Predicate");
        GT_STATIC_ASSERT((boost::mpl::size<typename esf_get_w_per_functor<EsfF, Pred>::type>::type::value == 1),
            "Each ESF should have a single output argument");
        typedef typename boost::mpl::at_c<typename esf_get_w_per_functor<EsfF>::type, 0>::type type;
    };

    /**
        If Pred derives from false_type, then `type` provide a mpl::vector of placeholders
        that corresponds to fields (temporary or not) that are read by EsfF.

        If Pred derives from true_type, then `type` provide a mpl::vector of pairs of
        placeholders and extents that corresponds to fields (temporary or not) that are
        read by EsfF.
     */
    template <typename EsfF, typename Pred = boost::false_type>
    struct esf_get_r_per_functor {
        GT_STATIC_ASSERT((is_esf_descriptor<EsfF>::value), "Wrong Type");
        GT_STATIC_ASSERT((is_meta_predicate<Pred>::type::value), "Not a Predicate");
        typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename EsfF::args_t>::type::value> range;
        typedef typename boost::mpl::fold<range,
            boost::mpl::vector0<>,
            boost::mpl::if_<typename is_written<EsfF>::template apply<boost::mpl::_2>,
                boost::mpl::_1,
                boost::mpl::push_back<boost::mpl::_1, esf_get_arg_at<EsfF, Pred, boost::mpl::_2>>>>::type type;
    };

    /**
       @brief It computes an associative sequence of all arg types specified by the user
        that are written into by at least one ESF
     */
    template <typename EsfSequence>
    struct compute_readwrite_args {
        GT_STATIC_ASSERT((is_sequence_of<EsfSequence, is_esf_descriptor>::value), "Wrong Type");
        typedef typename boost::mpl::fold<EsfSequence,
            boost::mpl::set0<>,
            copy_into_set<esf_get_w_per_functor<boost::mpl::_2>, boost::mpl::_1>>::type type;
    };

    /*
      Given an array of pairs (placeholder, extent) checks if all
      extents are the same and equal to the extent passed in
     */
    template <typename VectorOfPairs>
    struct check_all_horizotal_extents_are_zero {
        template <typename Pair>
        struct _check : bool_constant<Pair::second::iminus::value == 0 && Pair::second::iplus::value == 0 &&
                                      Pair::second::jminus::value == 0 && Pair::second::jplus::value == 0> {};

        typedef typename is_sequence_of<VectorOfPairs, _check>::type type;
    };

    namespace esf_metafunctions_impl_ {
        GT_META_LAZY_NAMESPACE {
            template <class Esf>
            struct tuple_from_esf {
                using type = std::tuple<Esf>;
            };
            template <class Esfs>
            struct tuple_from_esf<independent_esf<Esfs>> {
                using type = Esfs;
            };
        }
        GT_META_DELEGATE_TO_LAZY(tuple_from_esf, class Esf, Esf);
    } // namespace esf_metafunctions_impl_
    // Takes a list of esfs and independent_esf and produces a list of esfs, with the independent unwrapped
    template <class Esfs,
        class EsfLists = GT_META_CALL(meta::transform, (esf_metafunctions_impl_::tuple_from_esf, Esfs))>
    GT_META_DEFINE_ALIAS(unwrap_independent, meta::flatten, EsfLists);

} // namespace gridtools
