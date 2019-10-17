/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

/** \defgroup stencil-composition Stencil Composition
 */

/**
 *  @file
 *
 *  Here are the headers that contain the definitions of the entitles that are designed to be used directly by user
 *  for building stencil composition.
 *  The stuff that is needed only for definitions of the stencil functions should not be included here.
 */

#include "../storage/sid.hpp"
#include "accessor.hpp"
#include "caches/define_caches.hpp"
#include "esf.hpp"
#include "expressions/expressions.hpp"
#include "global_parameter.hpp"
#include "grid.hpp"
#include "make_computation.hpp"
#include "make_param_list.hpp"
#include "make_stage.hpp"
#include "make_stencils.hpp"

#include "backend_naive/entry_point.hpp"
#include "backend_x86/entry_point.hpp"

#ifndef GT_ICOSAHEDRAL_GRIDS
#include "backend_mc/entry_point.hpp"
#endif

#ifdef __CUDACC__
#include "backend_cuda/entry_point.hpp"
#endif
