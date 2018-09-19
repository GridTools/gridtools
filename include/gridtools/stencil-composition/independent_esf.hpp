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

#include <type_traits>

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/meta.hpp"
#include "./esf_fwd.hpp"

namespace gridtools {

    template <class>
    struct independent_esf;

    template <class T>
    GT_META_DEFINE_ALIAS(is_independent, meta::is_instantiation_of, (independent_esf, T));

    template <class Esfs>
    struct is_esf_descriptor<independent_esf<Esfs>> : std::true_type {};

    template <class Esfs>
    struct independent_esf {
        GRIDTOOLS_STATIC_ASSERT(
            (meta::all_of<is_esf_descriptor, Esfs>::value), "Error: independent_esf requires a sequence of esf's");
        // independent_esf always contains a flat list of esfs! No independent_esf inside.
        // This is ensured by make_independent design. That's why this assert is internal.
        GRIDTOOLS_STATIC_ASSERT((!meta::any_of<is_independent, Esfs>::value), GT_INTERNAL_ERROR);
        using esf_list = Esfs;
    };

} // namespace gridtools
