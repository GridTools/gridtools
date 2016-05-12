#pragma once
#include <memory>

#include "conditionals/fill_conditionals.hpp"
#include "../common/generic_metafunctions/vector_to_set.hpp"
#include "computation_grammar.hpp"
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

            typedef meta_array< mss_vector, boost::mpl::quote1< is_mss_descriptor > > type;
        };
    } // namespace _impl

    template < bool Positional, typename Backend, typename Domain, typename Grid, typename... Mss >
    std::shared_ptr< computation< typename _impl::reduction_helper< Mss... >::reduction_type_t > >
    make_computation_impl(Domain &domain, const Grid &grid, Mss... args_) {

        GRIDTOOLS_STATIC_ASSERT(
            (is_domain_type< Domain >::value), "syntax error in make_computation: invalid domain type");
        GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "syntax error in make_computation: invalid grid type");
        GRIDTOOLS_STATIC_ASSERT((accumulate(logical_and(), is_computation_token< Mss >::value...)),
            "syntax error in make_computation: invalid token");

        /* traversing also the subtrees of the control flow*/
        typedef typename boost::mpl::fold< boost::mpl::vector< Mss... >,
            boost::mpl::vector0<>,
            boost::mpl::if_< is_condition< boost::mpl::_2 >,
                                               construct_conditionals_set< boost::mpl::_1, boost::mpl::_2 >,
                                               boost::mpl::_1 > >::type conditionals_set_mpl_t;

        typedef typename vector_to_set< conditionals_set_mpl_t >::type conditionals_check_t;

        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< conditionals_check_t >::type::value ==
                                    boost::mpl::size< conditionals_set_mpl_t >::type::value),
            "Either you yoused the same switch_variable (or conditional) twice, or you used in the same computation "
            "two or more switch_variable (or conditional) with the same index. The index Id in "
            "condition_variable<Type, Id> (or conditional<Id>) must be unique to the computation, and can be used only "
            "in one switch_ statement.");

        typedef typename boost::fusion::result_of::as_set< conditionals_set_mpl_t >::type conditionals_set_t;
        conditionals_set_t conditionals_set_;

        fill_conditionals(conditionals_set_, args_...);

        return std::make_shared< intermediate< Backend,
            meta_array< typename meta_array_generator< boost::mpl::vector0<>, Mss... >::type,
                                                   boost::mpl::quote1< is_amss_descriptor > >,
            Domain,
            Grid,
            conditionals_set_t,
            typename _impl::reduction_helper< Mss... >::reduction_type_t,
            Positional > >(
            domain, grid, conditionals_set_, _impl::reduction_helper< Mss... >::extract_initial_value(args_...));
    }

    template < typename Backend, typename Domain, typename Grid, typename... Mss >
    std::shared_ptr< computation< typename _impl::reduction_helper< Mss... >::reduction_type_t > > make_computation(
        Domain &domain, const Grid &grid, Mss... args_) {
        return make_computation_impl< POSITIONAL_WHEN_DEBUGGING, Backend >(domain, grid, args_...);
    }

    template < typename Backend, typename Domain, typename Grid, typename... Mss >
    std::shared_ptr< computation< typename _impl::reduction_helper< Mss... >::reduction_type_t > >
    make_positional_computation(Domain &domain, const Grid &grid, Mss... args_) {
        return make_computation_impl< true, Backend >(domain, grid, args_...);
    }
}
