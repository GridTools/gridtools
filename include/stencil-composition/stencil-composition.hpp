#pragma once

#include "intermediate_metafunctions.hpp"
#include "stencil-composition/esf.hpp"
#include "stencil-composition/make_esf.hpp"
#include "stencil-composition/make_stencils.hpp"
#include "stencil-composition/make_computation.hpp"
#include "stencil-composition/stencil.hpp"
#include "stencil-composition/axis.hpp"

#include "../storage/expandable_parameters.hpp"
#include "expandable_parameters/make_computation_expandable.hpp"

#ifndef STRUCTURED_GRIDS
#include "stencil-composition/icosahedral_grids/grid.hpp"
#endif
