#pragma once

#include "global_accessor.hpp"

// TODOMEETING protect this with define and ifdef inside
// TODOMEETING inline namespaces to protect grid backends

#ifdef STRUCTURED_GRIDS
#include "stencil_composition/structured_grids/accessor_metafunctions.hpp"
#include "stencil_composition/structured_grids/accessor.hpp"
#else
#include "stencil_composition/icosahedral_grids/accessor_metafunctions.hpp"
#include "stencil_composition/icosahedral_grids/accessor.hpp"
#endif

#include "global_accessor.hpp"
