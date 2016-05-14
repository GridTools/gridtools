#pragma once

#ifdef STRUCTURED_GRIDS
#include "../structured_grids/backend_host/run_esf_functor_host.hpp"
#else
#include "../icosahedral_grids/backend_host/run_esf_functor_host.hpp"
#endif
