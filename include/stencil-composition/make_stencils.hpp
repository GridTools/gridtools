#pragma once
#include <boost/mpl/assert.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/push_back.hpp>


#include "common/defs.hpp"
#include "stencil-composition/esf.hpp"
#include "stencil-composition/mss.hpp"
#ifdef CXX11_ENABLED
#include "make_stencils_cxx11.hpp"
#else
#include "make_stencils_cxx03.hpp"
#endif
