/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "stencil-composition/stencil.hpp"
#include "cache_flusher.hpp"
#include "defs.hpp"

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
            // we run a first time the stencil, since if there is data allocation before by other codes, the first run of the stencil
            // is very slow (we dont know why). The flusher should make sure we flush the cache
            stencil->run();
            flusher.flush();

            stencil->reset_meter();
            for (uint_t t = 0; t < tsteps; ++t) {
                flusher.flush();
                stencil->run();
            }
            std::cout << stencil->print_meter() << std::endl;
        }
    };
}
