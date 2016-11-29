/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include <memory>
#include "cache_flusher.hpp"
#include "defs.hpp"
#include "stencil-composition/stencil.hpp"

namespace gridtools {

    struct benchmarker {

        static void run(
#ifdef CXX11_ENABLED
            std::shared_ptr< gridtools::stencil >
#else
#ifdef __CUDACC__
            gridtools::stencil *
#else
            boost::shared_ptr< gridtools::stencil >
#endif
#endif
                stencil,
            uint_t tsteps) {
            cache_flusher flusher(cache_flusher_size);
            // we run a first time the stencil, since if there is data allocation before by other codes, the first run
            // of the stencil
            // is very slow (we dont know why). The flusher should make sure we flush the cache
            stencil->run();
            flusher.flush();

            stencil->reset_meter();
            for (uint_t t = 0; t < tsteps; ++t) {
                flusher.flush();
                stencil->run();
            }

            double time = stencil->get_meter();
            std::ostringstream out;
            if (time < 0)
                out << "\t[s]\t"
                    << "NoName"
                    << "NO_TIMES_AVAILABLE";
            else
                out << "NoName"
                    << "\t[s]\t" << time;

            std::cout << out.str() << std::endl;
        }
    };
}
