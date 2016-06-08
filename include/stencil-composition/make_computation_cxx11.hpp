#pragma once
#include <memory>

#include "conditionals/fill_conditionals.hpp"
#include "../common/generic_metafunctions/vector_to_set.hpp"
#include "computation_grammar.hpp"
#include "make_computation_cxx11_impl.hpp"
#include "make_computation_helper_cxx11.hpp"

namespace gridtools {

    namespace _impl {
        /**
         * @brief metafunction that extracts a meta array with all the mss descriptors found in the Sequence of types
         * @tparam Sequence sequence of types that contains some mss descriptors
         */
        template < typename Sequence >
        struct get_mss_array {
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::is_sequence< Sequence >::value), "Internal Error: wrong type");

            typedef typename boost::mpl::fold< Sequence,
                boost::mpl::vector0<>,
                boost::mpl::eval_if< is_mss_descriptor< boost::mpl::_2 >,
                                                   boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 >,
                                                   boost::mpl::_1 > >::type mss_vector;

            typedef meta_array< mss_vector, boost::mpl::quote1< is_computation_token > > type;
        };
    } // namespace _impl

    /**TODO: use auto when C++14 becomes supported*/
    template < bool Positional, typename Backend, typename Domain, typename Grid, typename... Mss >
    std::shared_ptr< intermediate< Backend,
        meta_array< typename meta_array_generator< boost::mpl::vector0<>, Mss... >::type,
                                       boost::mpl::quote1< is_computation_token > >,
        Domain,
        Grid,
        typename _impl::create_conditionals_set< Domain, Grid, Mss... >::type,
        typename _impl::reduction_helper< Mss... >::reduction_type_t,
        Positional > >
    make_computation_impl(Domain &domain, const Grid &grid, Mss... args_) {
        typedef typename _impl::create_conditionals_set< Domain, Grid, Mss... >::type conditionals_set_t;

        conditionals_set_t conditionals_set_;

        fill_conditionals(conditionals_set_, args_...);

        return std::make_shared< intermediate< Backend,
            meta_array< typename meta_array_generator< boost::mpl::vector0<>, Mss... >::type,
                                                   boost::mpl::quote1< is_computation_token > >,
            Domain,
            Grid,
            conditionals_set_t,
            typename _impl::reduction_helper< Mss... >::reduction_type_t,
            Positional > >(
            domain, grid, conditionals_set_, _impl::reduction_helper< Mss... >::extract_initial_value(args_...));
    }

    template < typename Backend,
        typename Domain,
        typename Grid,
        typename... Mss,
        typename = typename std::enable_if< is_domain_type< Domain >::value >::type >
    auto make_computation(Domain &domain, const Grid &grid, Mss... args_)
        -> decltype(make_computation_impl< POSITIONAL_WHEN_DEBUGGING, Backend >(domain, grid, args_...)) {
        return make_computation_impl< POSITIONAL_WHEN_DEBUGGING, Backend >(domain, grid, args_...);
    }

    template < typename Backend,
        typename Domain,
        typename Grid,
        typename... Mss,
        typename = typename std::enable_if< is_domain_type< Domain >::value >::type >
    auto make_positional_computation(Domain &domain, const Grid &grid, Mss... args_)
        -> decltype(make_computation_impl< true, Backend >(domain, grid, args_...)) {
        return make_computation_impl< true, Backend >(domain, grid, args_...);
    }

    // user protections
    template < typename... Args >
    short_t make_computation(Args...) {
        GRIDTOOLS_STATIC_ASSERT((sizeof...(Args)), "the computation is malformed");
        return -1;
    }
}
