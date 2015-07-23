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
#include <stencil-composition/caches/define_caches.hpp>
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
        if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0)
            printf("PRE\n");

        eval(out()) = eval(in());
        if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0)
        printf("TTT %d %d %f %f %p %p\n", threadIdx.x, threadIdx.y, eval(out()), eval(in()), &(eval(out())),&(eval(in())));

    }
};

struct functor2 {
    typedef const accessor<0> in;
    typedef accessor<1> out;
    typedef boost::mpl::vector<in,out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {
        if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0)
            printf("PRE2\n");
        eval(out()) = eval(in());
        if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0)
        printf("HHH %d %d %f %f %p %p\n", threadIdx.x, threadIdx.y, eval(out()), eval(in()), &(eval(out())), &(eval(in())));

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
    const int halo_size=2;
    const int d1=32+halo_size*2;
    const int d2=32+halo_size*2;
    const int d3 = 6;
    uint_t di[5] = {halo_size, halo_size, halo_size, d1-halo_size, d1};
    uint_t dj[5] = {halo_size, halo_size, halo_size, d2-halo_size, d2};

    gridtools::coordinates<axis> coords(di, dj);
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

    storage_type in(d1, d2, d3, -8.5, "in");
    storage_type out(d1, d2, d3, 0.0, "out");

    typedef boost::mpl::vector3<p_in, p_out, p_buff> accessor_list;
    gridtools::domain_type<accessor_list> domain(boost::fusion::make_vector(&in, &out));

#ifdef __CUDACC__
    gridtools::computation* pstencil =
#else
        boost::shared_ptr<gridtools::computation> pstencil =
#endif
        make_computation<gridtools::BACKEND, layout_ijk_t>
        (
            make_mss // mss_descriptor
            (
                execute<forward>(),
                define_caches(cache<IJ, p_buff, cLocal>()),
                make_esf<functor1>(p_in(), p_buff()), // esf_descriptor
                make_esf<functor2>(p_buff(), p_out()) // esf_descriptor
            ),
            domain, coords
        );

    pstencil->ready();

    pstencil->steady();
    domain.clone_to_gpu();

    pstencil->run();

    pstencil->finalize();

#ifdef __CUDACC__
    out.data().update_cpu();
#endif

    for(int i = di[2]; i < di[3]; ++i )
    {
        for(int j = dj[2]; j < dj[3]; ++j )
        {
            for(int k = 0; k < d3; ++k )
            {
                if(out(i,j,k) != in(i,j,k)) std::cout << "PROBL " << i << " " << j << " " << k << " " << out(i,j,k) << std::endl;
            }
        }
    }
    ASSERT_TRUE(true);
}



