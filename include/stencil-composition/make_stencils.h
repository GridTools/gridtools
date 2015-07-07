#pragma once
#include <boost/mpl/assert.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/push_back.hpp>


#include "common/defs.h"
#include "stencil-composition/esf.h"
#include "stencil-composition/mss.h"
#ifdef CXX11_ENABLED
#include "make_stencils_cxx11.h"
#else
#include "make_stencils_cxx03.h"
#endif
