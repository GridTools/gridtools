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
#ifndef _TRANSLATE_H_
#define _TRANSLATE_H_

#include "../../common/layout_map_metafunctions.hpp"

namespace gridtools {

    template < int D, typename map = typename default_layout_map< 3 >::type >
    struct translate_t;

    //   template <>
    //   struct translate_t<2, layout_map<0,1,2> > {
    //     typedef layout_map<0,1,2> map_type;
    //     inline int operator()(int I, int J) {return (I+1)*3+J+1;}
    //   };

    template <>
    struct translate_t< 2, layout_map< 0, 1 > > {
        typedef layout_map< 0, 1 > map_type;
        inline int operator()(int I, int J) { return (I + 1) * 3 + J + 1; }
    };

    template <>
    struct translate_t< 2, layout_map< 1, 0 > > {
        typedef layout_map< 1, 0 > map_type;
        inline int operator()(int I, int J) { return (J + 1) * 3 + I + 1; }
    };

    template <>
    struct translate_t< 3, layout_map< 0, 1, 2 > > {
        typedef layout_map< 0, 1, 2 > map_type;
        inline int operator()(int I, int J, int K) { return (K + 1) * 9 + (J + 1) * 3 + I + 1; }
    };

    template <>
    struct translate_t< 3, layout_map< 2, 1, 0 > > {
        typedef layout_map< 2, 1, 0 > map_type;
        inline int operator()(int I, int J, int K) { return (I + 1) * 9 + (J + 1) * 3 + K + 1; }
    };

    template <>
    struct translate_t< 3, layout_map< 1, 2, 0 > > {
        typedef layout_map< 1, 2, 0 > map_type;
        inline int operator()(int I, int J, int K) { return (J + 1) * 9 + (I + 1) * 3 + K + 1; }
    };

    template <>
    struct translate_t< 3, layout_map< 0, 2, 1 > > {
        typedef layout_map< 0, 2, 1 > map_type;
        inline int operator()(int I, int J, int K) { return (K + 1) * 9 + (I + 1) * 3 + J + 1; }
    };

    template <>
    struct translate_t< 3, layout_map< 2, 0, 1 > > {
        typedef layout_map< 2, 0, 1 > map_type;
        inline int operator()(int I, int J, int K) { return (I + 1) * 9 + (K + 1) * 3 + J + 1; }
    };

    template <>
    struct translate_t< 3, layout_map< 1, 0, 2 > > {
        typedef layout_map< 1, 0, 2 > map_type;
        inline int operator()(int I, int J, int K) { return (J + 1) * 9 + (K + 1) * 3 + I + 1; }
    };
}

#endif
