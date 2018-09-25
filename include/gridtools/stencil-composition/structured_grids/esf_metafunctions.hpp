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
#include <boost/mpl/equal.hpp>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/meta.hpp"
#include "./esf.hpp"

namespace gridtools {

    template <typename Esf1, typename Esf2>
    struct esf_equal {
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor<Esf1>::value && is_esf_descriptor<Esf2>::value), GT_INTERNAL_ERROR);
        typedef static_bool<boost::is_same<typename Esf1::esf_function, typename Esf2::esf_function>::value &&
                            boost::mpl::equal<typename Esf1::args_t, typename Esf2::args_t>::value>
            type;
    };

    template <typename Esf>
    struct esf_arg_list {
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor<Esf>::value), GT_INTERNAL_ERROR);
        typedef typename Esf::esf_function::arg_list type;
    };

    /** Retrieve the extent in esf_descriptor_with_extents

       \tparam Esf The esf_descriptor that must be the one speficying the extent
    */
    template <typename Esf>
    struct esf_extent;

    template <typename ESF, typename Extent, typename ArgArray>
    struct esf_extent<esf_descriptor_with_extent<ESF, Extent, ArgArray>> {
        using type = Extent;
    };

    GT_META_LAZY_NAMESPASE {
        template <class Esf, class Args>
        struct esf_replace_args;
        template <class F, class OldArgs, class NewArgs>
        struct esf_replace_args<esf_descriptor<F, OldArgs>, NewArgs> {
            using type = esf_descriptor<F, NewArgs>;
        };
        template <class F, class Extent, class OldArgs, class NewArgs>
        struct esf_replace_args<esf_descriptor_with_extent<F, Extent, OldArgs>, NewArgs> {
            using type = esf_descriptor_with_extent<F, Extent, NewArgs>;
        };
    }
    GT_META_DELEGATE_TO_LAZY(esf_replace_args, (class Esf, class Args), (Esf, Args));

} // namespace gridtools
