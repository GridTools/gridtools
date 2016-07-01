#pragma once

#include <boost/mpl/assert.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/filter_view.hpp>
#include <boost/ref.hpp>

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>

#include "stencil-composition/backend.hpp"
#include "stencil-composition/esf.hpp"
#include "stencil-composition/mss_metafunctions.hpp"
#ifndef __CUDACC__
#include <boost/make_shared.hpp>
#endif
#include "intermediate.hpp"
#include "../common/meta_array.hpp"
#include "caches/define_caches.hpp"

#ifndef NDEBUG
#define POSITIONAL_WHEN_DEBUGGING true
#ifndef SUPPRESS_MESSAGES
#pragma message( \
    ">>\n>> In debug mode each computation is positional,\n>> so the loop indices can be queried from within\n>> the operator functions")
#endif
#else
#define POSITIONAL_WHEN_DEBUGGING false
#endif

#ifdef CXX11_ENABLED
#include "make_computation_cxx11.hpp"
#else
#include "make_computation_cxx03.hpp"
#endif
