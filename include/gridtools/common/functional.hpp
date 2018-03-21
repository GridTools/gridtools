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
/**
@file
@brief Unstructured collection of small generic purpose functors and related helpers.
*/

#include <utility>

namespace gridtools {
    /// Forward the args to constructor.
    template < typename T >
    struct ctor {
        template < typename... Args >
        T operator()(Args &&... args) const {
            return {std::forward< Args >(args)...};
        }

#ifndef BOOST_RESULT_OF_USE_DECLTYPE
        using result_type = T;
#endif
    };

    struct noop {
        template < typename... Args >
        void operator()(Args &&...) const {}

#ifndef BOOST_RESULT_OF_USE_DECLTYPE
        using result_type = void;
#endif
    };

    struct identity {
        template < typename Arg >
        Arg operator()(Arg &&arg) const {
            return arg;
        }

#ifndef BOOST_RESULT_OF_USE_DECLTYPE
        template < typename >
        struct result;
        template < typename Arg >
        struct result< identity(Arg &&) > {
            using type = Arg;
        };
#endif
    };
}
