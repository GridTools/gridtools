#pragma once

#include <boost/config.hpp>
#if defined(__CUDACC__) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
# define BOOST_NO_CXX11_RVALUE_REFERENCES
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/mpl/greater.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/timer/timer.hpp>
#include <boost/mpl/contains.hpp>
#ifdef GCL_MPI
#include <mpi.h>
#endif

#include "common/gt_assert.h"
#include "common/defs.h"
#include "common/host_device.h"
#include "common/array.h"
#include "common/layout_map.h"


// #include "stencil-composition/axis.h"
// #include "stencil-composition/make_stencils.h"
// #include "stencil-composition/arg_type.h"
// #include "stencil-composition/execution_types.h"
// #include "stencil-composition/domain_type.h"
// #include "stencil-composition/computation.h"
// #include "stencil-composition/intermediate.h"
// #include "stencil-composition/make_computation.h"
