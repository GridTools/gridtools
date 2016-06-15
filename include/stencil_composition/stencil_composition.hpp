#pragma once

#include "intermediate_metafunctions.hpp"
#include "stencil_composition/esf.hpp"
#include "stencil_composition/make_stage.hpp"
#include "stencil_composition/make_stencils.hpp"
#include "stencil_composition/make_computation.hpp"
#include "stencil_composition/stencil.hpp"
#include "stencil_composition/axis.hpp"
#include "stencil_composition/grid.hpp"
#include "stencil_composition/grid_traits.hpp"

#ifdef CXX11_ENABLED
#include "../storage/expandable_parameters.hpp"
#include "expandable_parameters/make_computation_expandable.hpp"
#else
#include "expandable_parameters/specializations.hpp"
#endif

#ifndef STRUCTURED_GRIDS
#include "stencil_composition/icosahedral_grids/grid.hpp"
#endif
