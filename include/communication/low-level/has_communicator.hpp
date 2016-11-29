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

#include "../GCL.hpp"
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>

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
