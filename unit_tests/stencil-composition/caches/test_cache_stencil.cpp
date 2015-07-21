/*
 * test_cache_stencil.cpp
 *
 *  Created on: Jul 21, 2015
 *      Author: cosuna
 */

#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <common/defs.hpp>
#include <stencil-composition/backend.hpp>
#include <stencil-composition/caches/cache_metafunctions.hpp>
#include <stencil-composition/interval.hpp>
#include <stencil-composition/make_computation.hpp>


using namespace gridtools;
using namespace enumtype;

// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
typedef gridtools::interval<level<0,-1>, level<1, 1> > axis;

struct functor1 {
    typedef const accessor<0> in;
    typedef accessor<1> out;
    typedef boost::mpl::vector<in,out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {
        eval(out()) = eval(in());
    }
};

#ifdef __CUDACC__
  #define BACKEND backend<Cuda, Block >
#else
  #define BACKEND backend<Host, Block >
#endif

typedef layout_map<2,1,0> layout_ijk_t;
typedef gridtools::BACKEND::storage_type<float_type, layout_ijk_t >::type storage_type;
typedef gridtools::BACKEND::temporary_storage_type<float_type, layout_ijk_t >::type tmp_storage_type;

typedef arg<0, storage_type> p_in;
typedef arg<1, storage_type> p_out;
typedef arg<2, tmp_storage_type> p_buff;

TEST(cache_stencil, ij_cache)
{
    storage_type in, out;

    typedef boost::mpl::vector3<p_in, p_out, p_buff> accessor_list;
    gridtools::domain_type<accessor_list> domain(boost::fusion::make_vector(&in, &out));

    const int halo_size=2;
    const int d1=32+halo_size*2;
    const int d2=32+halo_size*2;
    const int d3 = 6;
    uint_t di[5] = {halo_size, halo_size, halo_size, d1-halo_size, d1};
    uint_t dj[5] = {halo_size, halo_size, halo_size, d2-halo_size, d2};

    gridtools::coordinates<axis> coords(di, dj);
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

#ifdef __CUDACC__
    gridtools::computation* pstencils =
#else
        boost::shared_ptr<gridtools::computation> pstencil =
#endif
        gridtools::make_computation<gridtools::BACKEND, layout_ijk_t>
        (
            gridtools::make_mss // mss_descriptor
            (
                execute<forward>(),
                gridtools::make_esf<functor1>(p_in(), p_buff()), // esf_descriptor
                gridtools::make_esf<functor1>(p_buff(), p_out()) // esf_descriptor
            ),
            domain, coords
        );

//    typedef caches_used_by_esfs<esf_sequence_t, caches_t>::type caches_used_t;

//    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal<caches_used_t, boost::mpl::vector3<cache1_t, cache2_t, cache3_t> >::value), "WRONG");
//    ASSERT_TRUE(true);
}



