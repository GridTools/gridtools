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
            stencil->reset_meter();
            cache_flusher flusher(cache_flusher_size);
            for (uint_t t = 0; t < tsteps; ++t) {
                flusher.flush();
                stencil->run();
            }
            std::cout << stencil->print_meter() << std::endl;
        }
    };
}
