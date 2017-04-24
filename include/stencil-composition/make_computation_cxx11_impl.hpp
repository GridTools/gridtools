/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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

namespace gridtools {

    namespace _impl {

        template < typename Domain, typename Grid, typename... Mss >
        struct create_conditionals_set {
            GRIDTOOLS_STATIC_ASSERT(
                (is_aggregator_type< Domain >::value), "syntax error in make_computation: invalid domain type");
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
