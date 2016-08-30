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
