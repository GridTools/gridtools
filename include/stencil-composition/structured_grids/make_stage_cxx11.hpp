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

    /*!
       \fn mss_descriptor<...> make_esf(ExecutionEngine, esf1, esf2, ...)
       \brief Function to create a Multistage Stencil that can then be executed
       \param esf{i}  i-th Elementary Stencil Function created with make_esf or a list specified as independent ESFs
       created with make independent

       Use this function to create a multi-stage stencil computation
     */

    /*!
       \fn independent_esf<...> make_independent(esf1, esf2, ...)
       \brief Function to create a list of independent Elementary Styencil Functions

       \param esf{i}  (must be i>=2) The max{i} Elementary Stencil Functions in the argument list will be treated as
       independent

       Function to create a list of independent Elementary Styencil Functions. This is used to let the library compute
       tight bounds on blocks to be used by backends
     */

    template < typename ESF, typename... ExtraArgs >
    esf_descriptor< ESF, boost::mpl::vector< ExtraArgs... > > make_stage(ExtraArgs &&... /*args_*/) {
        GRIDTOOLS_STATIC_ASSERT((accumulate(logical_and(), is_arg< ExtraArgs >::value...)), "Malformed make_esf");
        return esf_descriptor< ESF, boost::mpl::vector< ExtraArgs... > >();
    }

    template < typename ESF, typename Staggering, typename... ExtraArgs >
    esf_descriptor< ESF, boost::mpl::vector< ExtraArgs... >, Staggering > make_stage(ExtraArgs &&... args_) {
        GRIDTOOLS_STATIC_ASSERT((accumulate(logical_and(), is_arg< ExtraArgs >::value...)), "Malformed make_esf");
        return esf_descriptor< ESF, boost::mpl::vector< ExtraArgs... >, Staggering >();
    }

} // namespace gridtools
