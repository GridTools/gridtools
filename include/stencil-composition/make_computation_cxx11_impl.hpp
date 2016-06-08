#pragma once

namespace gridtools {

    namespace _impl {

        template < typename Domain, typename Grid, typename... Mss >
        struct create_conditionals_set {
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
                "Either you yoused the same switch_variable (or conditional) twice, or you used in the same "
                "computation "
                "two or more switch_variable (or conditional) with the same index. The index Id in "
                "condition_variable<Type, Id> (or conditional<Id>) must be unique to the computation, and can be used "
                "only "
                "in one switch_ statement.");

            typedef typename boost::fusion::result_of::as_set< conditionals_set_mpl_t >::type type;
        };
    } // namespace _impl
} // namespace gridtools
