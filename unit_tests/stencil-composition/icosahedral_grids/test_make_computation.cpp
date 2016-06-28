/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
/*
 * test_computation.cpp
 *
 *  Created on: Mar 9, 2015
 *      Author: carlosos
 */

#define BOOST_NO_CXX11_RVALUE_REFERENCES

#include <gridtools.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/fusion/include/make_vector.hpp>

#include "gtest/gtest.h"

#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/aggregator_type.hpp>

#ifdef CXX11_ENABLED

using namespace gridtools;

namespace make_computation_test{

    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    using backend_t = backend< enumtype::Host, enumtype::icosahedral, enumtype::Block >;
    using icosahedral_topology_t = gridtools::icosahedral_topology<backend_t>;

    struct test_functor {
        using in = in_accessor< 0, icosahedral_topology_t::cells, extent< 1 > >;
        using arg_list = boost::mpl::vector1<in>;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {}
    };
}

#endif
