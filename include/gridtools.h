#pragma once

#include <boost/config.hpp>
#if defined(__CUDACC__) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
# define BOOST_NO_CXX11_RVALUE_REFERENCES
#endif

#include "common/gt_assert.h"
#include "common/defs.h"
#include "common/host_device.h"
#include "storage/base_storage.h"
#include "storage/storage.h"
#include "common/array.h"
#include "common/layout_map.h"
#include "stencil-composition/axis.h"
#include "stencil-composition/make_stencils.h"
#include "stencil-composition/arg_type.h"
#include "stencil-composition/execution_types.h"
#include "stencil-composition/domain_type.h"
#include "stencil-composition/computation.h"
#include "stencil-composition/intermediate.h"
#include "stencil-composition/make_computation.h"
