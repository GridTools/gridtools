#pragma once

#include "common/generic_metafunctions/variadic_to_vector.hpp"
#include "stencil-composition/mss_metafunctions.hpp"

namespace gridtools {

    /*!
       \fn esf_descriptor<ESF, ...> make_esf(plc1, plc2, plc3, ...)
       \brief Function to create a Elementary Stencil Function
       \param plc{i} placeholder which represents the i-th argument to the functor ESF

       Use this function to associate a stencil functor (stage) to
       arguments (actually, placeholders to arguments)
     */

    template <typename ESF, typename ... ExtraArgs>
    esf_descriptor<ESF, boost::mpl::vector<ExtraArgs ...> >
    make_esf( ExtraArgs&& ... /*args_*/){
#ifdef PEDANTIC // not valid for generic accessors (which is an exotic feature though)
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< typename ESF::arg_list >::type::value >= sizeof...(ExtraArgs)),
            "number of arugmantes declared for an ESF is larger than the placeholders passed to make_esf");
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< typename ESF::arg_list >::type::value <= sizeof...(ExtraArgs)),
            "number of arugmantes declared for an ESF is smaller than the placeholders passed to make_esf");
#endif
        return esf_descriptor<ESF, boost::mpl::vector<ExtraArgs ...> >();
    }

    template <typename ESF, typename Staggering, typename ... ExtraArgs>
    esf_descriptor<ESF, boost::mpl::vector<ExtraArgs ...>, Staggering >
    make_esf( ExtraArgs&& ... args_){
        return esf_descriptor<ESF, boost::mpl::vector<ExtraArgs ...>, Staggering >();
    }

} // namespace gridtools
