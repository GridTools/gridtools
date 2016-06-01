#pragma once

#include <boost/config.hpp>
#if defined(__CUDACC__) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
#define BOOST_NO_CXX11_RVALUE_REFERENCES
#endif

#include <boost/mpl/greater.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/at.hpp>
#include <boost/fusion/include/make_vector.hpp>
#ifndef __CUDACC__
#include <boost/timer/timer.hpp>
#endif
#include <boost/mpl/contains.hpp>
#include <boost/utility/enable_if.hpp>

#include <communication/GCL.hpp>

#include "common/defs.hpp"
#include "common/host_device.hpp"
#include "common/array.hpp"
#include "common/layout_map.hpp"
#include "common/pointer.hpp"
#include "common/pointer_metafunctions.hpp"
