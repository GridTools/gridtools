#pragma once
#include <memory>

#include "specializations.hpp"
#include "../conditionals/fill_conditionals.hpp"
#include "../../common/generic_metafunctions/vector_to_set.hpp"
#include "../computation_grammar.hpp"
#include "expand_factor.hpp"
#include "intermediate_expand.hpp"

/**
   @file make_computation specific for expandable parameters
*/

namespace gridtools {

    template < bool Positional, typename Backend, typename Expand, typename Domain, typename Grid, typename ReductionType, typename... Mss >
    std::shared_ptr< computation<ReductionType> > make_computation_expandable_impl(Expand /**/, Domain &domain, const Grid &grid, Mss... args_) {

        //doing type checks and defining the conditionals set
        typedef typename _impl::create_conditionals_set<Domain, Grid, Mss...>::type conditionals_set_t;

        conditionals_set_t conditionals_set_;

        fill_conditionals(conditionals_set_, args_...);

        return std::make_shared< intermediate_expand< Backend,
            meta_array< typename meta_array_generator< boost::mpl::vector0<>, Mss... >::type,
                                                   boost::mpl::quote1< is_mss_descriptor > >,
                                                      Domain,
                                                      Grid,
                                                      conditionals_set_t,
                                                      ReductionType,
                                                      Positional,
                                                      Expand> >(domain, grid, conditionals_set_);
    }

    template < typename Backend, typename Expand, typename Domain, typename Grid, typename ReductionType, typename... Mss, typename = typename std::enable_if<is_expand_factor<Expand>::value >::type >
    std::shared_ptr< computation<ReductionType> > make_computation(Expand /**/, Domain &domain, const Grid &grid, Mss... args_) {
        return make_computation_expandable_impl< POSITIONAL_WHEN_DEBUGGING, Backend >(Expand(), domain, grid, args_...);
    }

    template < typename Backend, typename Expand, typename Domain, typename Grid, typename ReductionType, typename... Mss, typename = typename std::enable_if<is_expand_factor<Expand>::value >::type >
    std::shared_ptr< computation<ReductionType> > make_positional_computation(Expand /**/, Domain &domain, const Grid &grid, Mss... args_) {
            return make_computation_expandable_impl< true, Backend >(Expand(), domain, grid, args_...);
    }
}
