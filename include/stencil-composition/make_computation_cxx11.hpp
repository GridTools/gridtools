/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
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

        boost::shared_ptr< conditionals_set_t > conditionals_set_(new conditionals_set_t());

        fill_conditionals(*conditionals_set_, args_...);

        std::cout << "filled true??? "
                  << boost::fusion::at_key< conditional< 4294967294U, 0U > >(*conditionals_set_).value() << std::endl;
        std::cout << "filled true??? "
                  << boost::fusion::at_key< conditional< 4294967295U, 0U > >(*conditionals_set_).value() << std::endl;
        std::cout << "filled true??? "
                  << boost::fusion::at_key< conditional< 4294967295U, 1U > >(*conditionals_set_).value() << std::endl;
        std::cout << "filled true??? "
                  << boost::fusion::at_key< conditional< 4294967294U, 2U > >(*conditionals_set_).value() << std::endl;
        std::cout << "filled true??? "
                  << boost::fusion::at_key< conditional< 4294967295U, 2U > >(*conditionals_set_).value() << std::endl;

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
        typename = typename std::enable_if< is_aggregator_type< Domain >::value >::type >
    auto make_computation(Domain &domain, const Grid &grid, Mss... args_)
        -> decltype(make_computation_impl< POSITIONAL_WHEN_DEBUGGING, Backend >(domain, grid, args_...)) {
        return make_computation_impl< POSITIONAL_WHEN_DEBUGGING, Backend >(domain, grid, args_...);
    }

    template < typename Backend,
        typename Domain,
        typename Grid,
        typename... Mss,
        typename = typename std::enable_if< is_aggregator_type< Domain >::value >::type >
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
