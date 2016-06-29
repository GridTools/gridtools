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
#pragma once

#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
#include "../GCL.hpp"

namespace gridtools {

    template < typename _grid_ >
    struct has_communicator {
        typedef typename boost::is_same< typename _grid_::has_communicator, boost::true_type >::type type;
    };

#ifndef _GCL_MPI_
#define MPI_Comm int
#endif

    template < typename _grid_ >
    MPI_Comm get_communicator(
        _grid_ const &g, typename boost::enable_if< typename has_communicator< _grid_ >::type >::type * = 0) {
        return g.communicator();
    }

    template < typename _grid_ >
    MPI_Comm get_communicator(
        _grid_ const &g, typename boost::disable_if< typename has_communicator< _grid_ >::type >::type * = 0) {
        return gridtools::GCL_WORLD;
    }

#ifndef _GCL_MPI_
#undef MPI_Comm
#endif
}
