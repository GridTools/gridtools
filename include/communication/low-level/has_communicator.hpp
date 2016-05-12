#pragma once

#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
#include "../GCL.hpp"

namespace gridtools {

    template <typename _grid_>
    struct has_communicator {
        typedef typename boost::is_same<typename _grid_::has_communicator, boost::true_type>::type type;
    };

#ifndef _GCL_MPI_
#define MPI_Comm int
#endif

    template <typename _grid_>
    MPI_Comm get_communicator(_grid_ const& g, typename boost::enable_if<typename has_communicator<_grid_>::type >::type* =0) {
        return g.communicator();
    }

    template <typename _grid_>
    MPI_Comm get_communicator(_grid_ const& g, typename boost::disable_if<typename has_communicator<_grid_>::type >::type* =0) {
        return gridtools::GCL_WORLD;
    }

#ifndef _GCL_MPI_
#undef MPI_Comm
#endif

}
