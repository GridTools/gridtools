#pragma once

#include <stencil-composition/stencil-composition.hpp>
#include "horizontal_diffusion_repository.hpp"
#include "cache_flusher.hpp"
#include "defs.hpp"
#include <tools/verifier.hpp>

#ifdef USE_PAPI_WRAP
#include <papi_wrap.hpp>
#include <papi.hpp>
#endif

/**
  @file
  This file shows an implementation of the "horizontal diffusion" stencil, similar to the one used in COSMO
 */

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

//Temporary disable the expressions, as they are intrusive. The operators +,- are overloaded
//  for any type, which breaks most of the code after using expressions
#ifdef CXX11_ENABLED
using namespace expressions;
#endif

namespace horizontal_diffusion{
// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval<level<0,-1>, level<1,-1> > x_lap;
typedef gridtools::interval<level<0,-1>, level<1,-1> > x_flx;
typedef gridtools::interval<level<0,-1>, level<1,-1> > x_out;

typedef gridtools::interval<level<0,-2>, level<1,3> > axis;

// These are the stencil operators that compose the multistage stencil in this test
struct wlap_function {
    typedef accessor<0, enumtype::inout> out;
    typedef accessor<1, enumtype::in, extent<-1, 1, -1, 1> > in;
    typedef accessor<2, enumtype::in > crlato;
    typedef accessor<3, enumtype::in > crlatu;

    typedef boost::mpl::vector<out, in, crlato, crlatu> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_lap) {
        eval(out()) = eval(in(1,0,0)) + eval(in(-1,0,0)) - (gridtools::float_type)2*eval(in()) +
            eval(crlato()) * (eval(in(0,1,0)) - eval(in())) +
            eval(crlatu()) * (eval(in(0,-1,0)) - eval(in())) ;
    }
};

struct divflux_function {

    typedef accessor<0, enumtype::inout> out;
    typedef accessor<1, enumtype::in> in;
    typedef accessor<2, enumtype::in, extent<-1, 1, -1, 1> > lap;
    typedef accessor<3, enumtype::in> crlato;
    typedef accessor<4, enumtype::in> coeff;

    typedef boost::mpl::vector<out, in, lap, crlato, coeff> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_flx) {
        gridtools::float_type fluxx = eval(lap(1,0,0)) - eval(lap());
        gridtools::float_type fluxx_m = eval(lap(0,0,0)) - eval(lap(-1,0,0));

        gridtools::float_type fluxy = eval(crlato())*(eval(lap(0,1,0)) - eval(lap()));
        gridtools::float_type fluxy_m = eval(crlato())*(eval(lap(0,0,0)) - eval(lap(0,-1,0)));

        eval(out()) = eval(in())+ ((fluxx_m - fluxx) + (fluxy_m - fluxy))*eval(coeff());
    }
};

/*
 * The following operators and structs are for debugging only
 */
std::ostream& operator<<(std::ostream& s, wlap_function const) {
    return s << "wlap_function";
}
std::ostream& operator<<(std::ostream& s, divflux_function const) {
    return s << "flx_function";
}

bool test(uint_t x, uint_t y, uint_t z, uint_t t_steps)
{

    cache_flusher flusher(cache_flusher_size);

    uint_t d1 = x;
    uint_t d2 = y;
    uint_t d3 = z;
    uint_t halo_size = 2;

#ifdef CUDA_EXAMPLE
    #define BACKEND backend<Cuda, Block >
#else
#ifdef BACKEND_BLOCK
    #define BACKEND backend<Host, Block >
#else
    #define BACKEND backend<Host, Naive >
#endif
#endif


    typedef horizontal_diffusion::repository::storage_type storage_type;
    typedef horizontal_diffusion::repository::j_storage_type j_storage_type;
    typedef horizontal_diffusion::repository::tmp_storage_type tmp_storage_type;

    horizontal_diffusion::repository repository(d1, d2, d3, halo_size);
    repository.init_fields();

    repository.generate_reference_simple();

     // Definition of the actual data fields that are used for input/output
    storage_type& in = repository.in();
    storage_type& out = repository.out();
    storage_type& coeff = repository.coeff();
    j_storage_type& crlato = repository.crlato();
    j_storage_type& crlatu = repository.crlatu();

    // Definition of placeholders. The order of them reflect the order the user will deal with them
    // especially the non-temporary ones, in the construction of the domain
    typedef arg<0, tmp_storage_type > p_lap;
    typedef arg<1, storage_type > p_coeff;
    typedef arg<2, storage_type > p_in;
    typedef arg<3, storage_type > p_out;
    typedef arg<4, j_storage_type > p_crlato;
    typedef arg<5, j_storage_type > p_crlatu;


    // An array of placeholders to be passed to the domain
    typedef boost::mpl::vector<p_lap, p_coeff, p_in, p_out, p_crlato, p_crlatu> accessor_list;

    // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
    // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
#if defined( CXX11_ENABLED )
    gridtools::domain_type<accessor_list> domain( (p_out() = out), (p_in() = in), (p_coeff() = coeff),
                                                  (p_crlato() = crlato), (p_crlatu() = crlatu));
#else
    gridtools::domain_type<accessor_list> domain(boost::fusion::make_vector(&coeff, &in, &out, &crlato, &crlatu));
#endif
    // Definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions,
    // while the vertical ones are set according the the axis property soon after
    // gridtools::grid<axis> grid(2,d1-2,2,d2-2);
    uint_t di[5] = {halo_size, halo_size, halo_size, d1-halo_size-1, d1};
    uint_t dj[5] = {halo_size, halo_size, halo_size, d2-halo_size-1, d2};

    gridtools::grid<axis> grid(di, dj);
    grid.value_list[0] = 0;
    grid.value_list[1] = d3-1;

// \todo simplify the following using the auto keyword from C++11
#ifdef __CUDACC__
    gridtools::computation* simple_hori_diff =
#else
        boost::shared_ptr<gridtools::computation> simple_hori_diff =
#endif
        gridtools::make_computation<gridtools::BACKEND>
        (
            gridtools::make_mss // mss_descriptor
            (
                execute<forward>(),
                define_caches(cache<IJ, p_lap, local>()),
                gridtools::make_esf<wlap_function>(p_lap(), p_in(), p_crlato(), p_crlatu()), // esf_descriptor
                gridtools::make_esf<divflux_function>(p_out(), p_in(), p_lap(), p_crlato(), p_coeff())
            ),
            domain, grid
        );

    simple_hori_diff->ready();

    simple_hori_diff->steady();

    for(uint_t t=0; t < t_steps; ++t){
        flusher.flush();
        simple_hori_diff->run();
    }

    repository.update_cpu();

#ifdef CXX11_ENABLED
    verifier verif(1e-13);
    array<array<uint_t, 2>, 3> halos{{ {halo_size, halo_size}, {halo_size,halo_size}, {halo_size,halo_size} }};
    bool result = verif.verify(grid, repository.out_ref(), repository.out(), halos);
#else
    verifier verif(1e-13, halo_size);
    bool result = verif.verify(grid, repository.out_ref(), repository.out());
#endif

    if(!result){
        std::cout << "ERROR"  << std::endl;
    }

#ifdef BENCHMARK
    for(uint_t t=1; t < t_steps; ++t){
        flusher.flush();
        simple_hori_diff->run();
    }
    simple_hori_diff->finalize();
    std::cout << simple_hori_diff->print_meter() << std::endl;
#endif

  return result; /// lapse_time.wall<5000000 &&
}

}//namespace simple_hori_diff
