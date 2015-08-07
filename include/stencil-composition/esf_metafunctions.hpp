#pragma once

#include <boost/mpl/contains.hpp>
#include "stencil-composition/esf.hpp"

namespace gridtools {

/** Metafunction checking if an ESF has, as argument, a given placeholder
*/
template<typename Arg>
struct esf_has_parameter_h{
    template<typename Esf>
    struct apply{
        typedef typename boost::mpl::contains<typename Esf::args_t, Arg>::type type;
    };
};

/**
   Given an ESF this metafunction provides the list of placeholders (if Pred derives
   from false_type), or map between placeholders in this ESF and the ranges
   associated with it (if Pred derives from true_type)
 */
    template<typename Esf, typename Pred=boost::false_type>
struct esf_args
{
    GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor<Esf>::value), "Wrong Type");

    typedef typename boost::mpl::if_<
        Pred,
        typename Esf::args_with_ranges,
        typename Esf::args_t>::type type;
};

/**
   Given an ESF this metafunction provides the placeholder (if Pred derives
   from false_type) at a given index in the list of placeholders, or mpl::pair of
   placeholder and range (if Pred derives from true_type)
 */
template <typename Esf, typename Pred>
struct esf_get_arg_at {
    template <typename Index>
    struct apply {
        typedef typename boost::mpl::if_<
            Pred,
            typename boost::mpl::at<typename Esf::args_with_ranges, typename boost::mpl::at<typename Esf::args_t, Index>::type>::type,
            typename boost::mpl::at<typename Esf::args_t, Index>::type
            >::type type;
    };
};


/** Provide true_type if the placeholder, which index is Index in the list of placeholders of
    Esf, corresponds to a temporary that is written.
 */
template <typename Esf>
struct is_written_temp {
    template <typename Index>
    struct apply {
        typedef typename boost::mpl::if_<
            is_plchldr_to_temp<typename boost::mpl::at<typename Esf::args_t, Index>::type>,
            typename boost::mpl::if_<
                boost::is_const<typename boost::mpl::at<typename Esf::esf_function::arg_list, Index>::type>,
                boost::false_type,
                boost::true_type
            >::type,
            boost::false_type
        >::type type;
    };
};

/** Provide true_type if the placeholder, which index is Index in the list of placeholders of
    Esf, correspond to a field (temporary or not) that is is written.
 */
template <typename Esf>
struct is_written {
    template <typename Index>
    struct apply {
        typedef typename boost::mpl::if_<
            is_plchldr<typename boost::mpl::at<typename Esf::args_t, Index>::type>,
            typename boost::mpl::if_<
                boost::is_const<typename boost::mpl::at<typename Esf::esf_function::arg_list, Index>::type>,
                boost::false_type,
                boost::true_type
            >::type,
            boost::false_type
        >::type type;
    };
};

/**
    If Pred derives from false_type, ::type provide a mpl::vector of placeholders
    that corresponds to temporary fields that are written by EsfF.

    If Pred derives from true_type, ::type provide a mpl::vector of pairs of
    placeholders and ranges that corresponds to temporary fields that are written by EsfF.
 */
template <typename EsfF, typename Pred = boost::false_type>
struct esf_get_w_temps_per_functor {
    typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename EsfF::args_t>::type::value> range;
    typedef typename boost::mpl::fold<
        range,
        boost::mpl::vector0<>,
        boost::mpl::if_<
            typename is_written_temp<EsfF>::template apply<boost::mpl::_2>,
            boost::mpl::push_back<
                boost::mpl::_1,
                typename esf_get_arg_at<EsfF, Pred>::template apply<boost::mpl::_2>
            >,
            boost::mpl::_1
        >
    >::type type;
};

/**
    If Pred derives from false_type, ::type provide a mpl::vector of placeholders
    that corresponds to fields that are read by EsfF.

    If Pred derives from true_type, ::type provide a mpl::vector of pairs of
    placeholders and ranges that corresponds to fields that are read by EsfF.
 */
template <typename EsfF, typename Pred = boost::false_type>
struct esf_get_r_temps_per_functor {
    typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename EsfF::args_t>::type::value> range;
    typedef typename boost::mpl::fold<
        range,
        boost::mpl::vector0<>,
        boost::mpl::if_<
            typename is_written_temp<EsfF>::template apply<boost::mpl::_2>,
            boost::mpl::_1,
            boost::mpl::push_back<
                boost::mpl::_1,
                typename esf_get_arg_at<EsfF, Pred>::template apply<boost::mpl::_2>
            >
        >
    >::type type;
};

/**
    If Pred derives from false_type, ::type provide a mpl::vector of placeholders
    that corresponds to fields (temporary or not) that are written by EsfF.

    If Pred derives from true_type, ::type provide a mpl::vector of pairs of
    placeholders and ranges that corresponds to fields (temporary or not) that are
    written by EsfF.
 */
template <typename EsfF, typename Pred = boost::false_type>
struct esf_get_w_per_functor {
    typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename EsfF::args_t>::type::value> range;
    typedef typename boost::mpl::fold<
        range,
        boost::mpl::vector0<>,
        boost::mpl::if_<
            typename is_written<EsfF>::template apply<boost::mpl::_2>,
            boost::mpl::push_back<
                boost::mpl::_1,
                typename esf_get_arg_at<EsfF, Pred>::template apply<boost::mpl::_2>
            >,
            boost::mpl::_1
        >
    >::type type;
};

/**
   If the ESF stencil operator writes only one parameter (temporary or
   not) corresponding to a placeholder, it returns this placeholder,
   otherwise it returns the first placeholder to a field that is
   written (temporary or not).
 */
template <typename EsfF, typename Pred = boost::false_type>
struct esf_get_the_only_w_per_functor {
    GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<typename esf_get_w_per_functor<EsfF, Pred>::type>::type::value == 0),
                            "Each ESF should have a single output argument");
    typedef typename boost::mpl::at_c<typename esf_get_w_per_functor<EsfF>::type, 0>::type type;
};

/**
    If Pred derives from false_type, ::type provide a mpl::vector of placeholders
    that corresponds to fields (temporary or not) that are read by EsfF.

    If Pred derives from true_type, ::type provide a mpl::vector of pairs of
    placeholders and ranges that corresponds to fields (temporary or not) that are
    read by EsfF.
 */
template <typename EsfF, typename Pred = boost::false_type>
struct esf_get_r_per_functor {
    typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename EsfF::args_t>::type::value> range;
    typedef typename boost::mpl::fold<
        range,
        boost::mpl::vector0<>,
        boost::mpl::if_<
            typename is_written<EsfF>::template apply<boost::mpl::_2>,
            boost::mpl::_1,
            boost::mpl::push_back<
                boost::mpl::_1,
                typename esf_get_arg_at<EsfF, Pred>::template apply<boost::mpl::_2>
            >
        >
    >::type type;
};


} //namespace gridtools
