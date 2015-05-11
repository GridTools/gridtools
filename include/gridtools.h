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
#include <boost/mpl/at.hpp>
#include <boost/fusion/include/make_vector.hpp>
#ifndef __CUDACC__
#include <boost/timer/timer.hpp>
#endif
#include <boost/mpl/contains.hpp>
#include <boost/utility/enable_if.hpp>

#include<communication/GCL.h>

#include "common/gt_assert.h"
#include "common/defs.h"
#include "common/host_device.h"
#include "common/array.h"
#include "common/layout_map.h"
#include "common/gridtools_runtime.h"
