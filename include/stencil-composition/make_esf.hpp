#pragma once

#ifdef STRUCTURED_GRIDS
#ifdef CXX11_ENABLED
#include "stencil-composition/structured_grids/make_esf_cxx11.hpp"
#else
#include "stencil-composition/structured_grids/make_esf_cxx03.hpp"
#endif
#else
#include "stencil-composition/icosahedral_grids/make_esf.hpp"
#endif
