#pragma once

#include "common/generic_metafunctions/variadic_to_vector.hpp"
#include "mss_metafunctions.hpp"

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
       \param esf{i}  i-th Elementary Stencil Function created with make_esf or a list specified as independent ESFs created with make independent

       Use this function to create a multi-stage stencil computation
     */

    /*!
       \fn independent_esf<...> make_independent(esf1, esf2, ...)
       \brief Function to create a list of independent Elementary Styencil Functions

       \param esf{i}  (must be i>=2) The max{i} Elementary Stencil Functions in the argument list will be treated as independent

       Function to create a list of independent Elementary Styencil Functions. This is used to let the library compute tight bounds on blocks to be used by backends
     */

    template <typename ESF, typename ... ExtraArgs>
    esf_descriptor<ESF, boost::mpl::vector<ExtraArgs ...> >
    make_esf( ExtraArgs&& ... /*args_*/){
        return esf_descriptor<ESF, boost::mpl::vector<ExtraArgs ...> >();
    }

    template <typename ESF, typename Staggering, typename ... ExtraArgs>
    esf_descriptor<ESF, boost::mpl::vector<ExtraArgs ...>, Staggering >
    make_esf( ExtraArgs&& ... args_){
        return esf_descriptor<ESF, boost::mpl::vector<ExtraArgs ...>, Staggering >();
    }

    template <typename ExecutionEngine,
        typename ... MssParameters >
    mss_descriptor<
        ExecutionEngine,
        typename extract_mss_esfs<typename variadic_to_vector<MssParameters ... >::type >::type,
        typename extract_mss_caches<typename variadic_to_vector<MssParameters ...>::type >::type
    >
    make_mss(ExecutionEngine&& /**/, MssParameters ...  ) {
        return mss_descriptor<
            ExecutionEngine,
            typename extract_mss_esfs<typename variadic_to_vector<MssParameters ... >::type >::type,
            typename extract_mss_caches<typename variadic_to_vector<MssParameters ... >::type >::type
        >();
    }


    template <typename ... EsfDescr >
    independent_esf< boost::mpl::vector<EsfDescr ...> >
    make_independent(EsfDescr&& ... ) {
        return independent_esf<boost::mpl::vector<EsfDescr... > >();
    }

} // namespace gridtools
