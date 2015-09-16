#pragma once

#include <boost/mpl/contains.hpp>
#include "stencil-composition/esf.hpp"
#include "stencil-composition/accessor.hpp"

namespace gridtools {

// Metafunctions
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

template<typename Arg>
struct esf_has_parameter_h{
    template<typename Esf>
    struct apply{
        typedef typename boost::mpl::contains<typename Esf::args_t, Arg>::type type;
    };
};

template<typename Esf>
struct esf_args
{
    GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor<Esf>::value), "Wrong Type");
    typedef typename Esf::args_t type;
};

template <typename Esf>
struct esf_get_arg_index {
    template <typename Index>
    struct apply {
        typedef typename boost::mpl::at<typename Esf::args_t, Index>::type type;
    };
};

template <typename EsfF>
struct esf_get_temps_per_functor {
    typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename EsfF::args_t>::type::value> range;
    typedef typename boost::mpl::fold<
        range,
        boost::mpl::vector<>,
        boost::mpl::if_<
            typename is_written_temp<EsfF>::template apply<boost::mpl::_2>,
            boost::mpl::push_back<
                boost::mpl::_1,
                typename esf_get_arg_index<EsfF>::template apply<boost::mpl::_2>
            >,
            boost::mpl::_1
        >
    >::type type;
};


} //namespace gridtools

