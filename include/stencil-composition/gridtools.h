#pragma once

#include <boost/config.hpp>
#if defined(__CUDACC__) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
# define BOOST_NO_CXX11_RVALUE_REFERENCES
#endif

#include "gt_assert.h"
#include "defs.h"
#include "host_device.h"
#include "storage.h"
#include "cuda_storage.h"
#include "array.h"
#include "layout_map.h"
#include "axis.h"
#include "make_stencils.h"
#include "arg_type.h"
#include "execution_types.h"
#include "domain_type.h"
#include "computation.h"
#include "intermediate.h"
#include "make_computation.h"
