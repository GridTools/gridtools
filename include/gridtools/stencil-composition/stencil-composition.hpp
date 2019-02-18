/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
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

#include "accessor.hpp"
#include "backend.hpp"
#include "caches/define_caches.hpp"
#include "computation.hpp"
#include "esf.hpp"
#include "global_accessor.hpp"
#include "grid.hpp"
#include "make_computation.hpp"
#include "make_stage.hpp"
#include "make_stencils.hpp"
#include "storage_info_extender.hpp"
