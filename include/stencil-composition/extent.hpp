#pragma once

#include "./empty_extent.hpp"

#ifdef STRUCTURED_GRIDS
#include "structured_grids/extent.hpp"
#else
#include "icosahedral_grids/extent.hpp"
#endif

#include "./empty_extent_specializations.hpp"
