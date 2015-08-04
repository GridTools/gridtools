#pragma once

#include <boost/mpl/contains.hpp>
#include <stencil-composition/esf.hpp>

namespace gridtools {

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

};

