#pragma once
#include "iterate_domain_impl.hpp"

#ifdef STRUCTURED_GRIDS
#include "structured_grids/iterate_domain.hpp"
#else
#include "icosahedral_grids/iterate_domain.hpp"
#endif
