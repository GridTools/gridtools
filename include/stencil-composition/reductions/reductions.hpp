#pragma once

#include "../esf.hpp"
#include "../make_stage.hpp"
#ifdef CXX11_ENABLE
#include "make_reduction_cxx11.hpp"
#else
#include "make_reduction_cxx03.hpp"
#endif

#include "../make_computation.hpp"
#include "../axis.hpp"
#include "../../common/binops.hpp"
