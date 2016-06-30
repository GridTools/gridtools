#pragma once

#ifdef STRUCTURED_GRIDS
#ifdef CXX11_ENABLED
#include "stencil-composition/structured_grids/make_stage_cxx11.hpp"
#else
#include "stencil-composition/structured_grids/make_stage_cxx03.hpp"
#endif
#else
#include "stencil-composition/icosahedral_grids/make_stage.hpp"
#endif
