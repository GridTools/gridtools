/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/is_sequence_of.hpp"
#include "../meta/is_instantiation_of.hpp"
#include "../meta/logical.hpp"
#include "../meta/macros.hpp"
#include "caches/cache_traits.hpp"
#include "esf.hpp"
#include "execution_types.hpp"

/**
@file
@brief descriptor of the Multi Stage Stencil (MSS)
*/
namespace gridtools {
    /** @brief Descriptors for  Multi Stage Stencil (MSS) */
    template <typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence = std::tuple<>>
    struct mss_descriptor {
        GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_esf_descriptor, EsfDescrSequence>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<EsfDescrSequence, is_esf_descriptor>::value), GT_INTERNAL_ERROR);

        GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_cache, CacheSequence>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<CacheSequence, is_cache>::value), GT_INTERNAL_ERROR);

        GRIDTOOLS_STATIC_ASSERT(is_execution_engine<ExecutionEngine>::value, GT_INTERNAL_ERROR);

        using execution_engine_t = ExecutionEngine;
        using esf_sequence_t = EsfDescrSequence;
        using cache_sequence_t = CacheSequence;
    };

    template <typename Mss>
    GT_META_DEFINE_ALIAS(is_mss_descriptor, meta::is_instantiation_of, (mss_descriptor, Mss));
} // namespace gridtools
