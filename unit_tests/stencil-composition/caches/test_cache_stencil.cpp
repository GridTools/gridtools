#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include "common/defs.hpp"
#include "stencil-composition/stencil-composition.hpp"
#include <tools/verifier.hpp>

namespace test_cache_stencil {

using namespace gridtools;
using namespace enumtype;

// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1, 1> > axis;

struct functor1 {
    typedef accessor<0, enumtype::in, extent<-1,1,-1,1> > in;
    typedef accessor<1, enumtype::inout> out;
    typedef boost::mpl::vector<in,out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {
        eval(out()) = eval(in());
    }
};

struct functor2 {
    typedef accessor<0, enumtype::in, extent<-1,1,-1,1> > in;
    typedef accessor<1, enumtype::inout> out;
    typedef boost::mpl::vector<in,out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {
        eval(out()) = (eval(in(-1,0,0)) + eval(in(1,0,0)) + eval(in(0,-1,0)) + eval(in(0,1,0))) / (float_type)4.0 ;
    }
};

#ifdef __CUDACC__
  #define BACKEND backend<Cuda, Block >
#else
  #define BACKEND backend<Host, Block >
#endif

typedef layout_map<2,1,0> layout_ijk_t;
    typedef gridtools::BACKEND::storage_type<float_type, storage_info<0,layout_ijk_t> >::type storage_type;
    typedef gridtools::BACKEND::temporary_storage_type<float_type, storage_info<0,layout_ijk_t> >::type tmp_storage_type;

typedef arg<0, storage_type> p_in;
typedef arg<1, storage_type> p_out;
typedef arg<2, tmp_storage_type> p_buff;

}

using namespace gridtools;
using namespace enumtype;
using namespace test_cache_stencil;

class cache_stencil : public ::testing::Test
{
protected:

    const uint_t m_halo_size;
    const uint_t m_d1, m_d2, m_d3;

    array<uint_t, 5> m_di, m_dj;

    gridtools::grid<axis> m_grid;
    typename storage_type::meta_data_t m_meta;
    storage_type m_in, m_out;

    cache_stencil() :
        m_halo_size(2), m_d1(32+m_halo_size), m_d2(32+m_halo_size), m_d3(6),
#ifdef CXX11_ENABLED
        m_di{m_halo_size, m_halo_size, m_halo_size, m_d1-m_halo_size-1, m_d1},
        m_dj{m_halo_size, m_halo_size, m_halo_size, m_d2-m_halo_size-1, m_d2},
#else
        m_di(m_halo_size, m_halo_size, m_halo_size, m_d1-m_halo_size-1, m_d1),
        m_dj(m_halo_size, m_halo_size, m_halo_size, m_d2-m_halo_size-1, m_d2),
#endif
        m_grid(m_di, m_dj),
        m_meta(m_d1, m_d2, m_d3),
        m_in(m_meta, -8.5, "in"),
        m_out(m_meta, 0.0, "out")
    {
        m_grid.value_list[0] = 0;
        m_grid.value_list[1] = m_d3-1;
    }

    virtual void SetUp()
    {
        for(int i = m_di[2]; i < m_di[3]; ++i )
        {
            for(int j = m_dj[2]; j < m_dj[3]; ++j )
            {
                for(int k = 0; k < m_d3; ++k )
                {
                    m_in(i,j,k) = i+j*100+k*10000;
                }
            }
        }
    }
};

TEST_F(cache_stencil, ij_cache)
{
    typedef boost::mpl::vector3<p_in, p_out, p_buff> accessor_list;
    gridtools::domain_type<accessor_list> domain(boost::fusion::make_vector(&m_in, &m_out));

#ifdef __CUDACC__
    gridtools::computation* pstencil =
#else
        boost::shared_ptr<gridtools::computation> pstencil =
#endif
        make_computation<gridtools::BACKEND>
        (
            make_mss // mss_descriptor
            (
                execute<forward>(),
                define_caches(cache<IJ, p_buff, local>()),
                make_esf<functor1>(p_in(), p_buff()), // esf_descriptor
                make_esf<functor1>(p_buff(), p_out()) // esf_descriptor
            ),
            domain, m_grid
        );

    pstencil->ready();

    pstencil->steady();

    pstencil->run();

    pstencil->finalize();

#ifdef __CUDACC__
    m_out.data().update_cpu();
#endif

#ifdef CXX11_ENABLED
    verifier verif(1e-13);
    array<array<uint_t, 2>, 3> halos{{ {m_halo_size,m_halo_size}, {m_halo_size,m_halo_size}, {m_halo_size,m_halo_size} }};
    ASSERT_TRUE(verif.verify(m_in, m_out, halos) );
#else
    verifier verif(1e-13, m_halo_size);
    ASSERT_TRUE(verif.verify(m_in, m_out) );
#endif
}

TEST_F(cache_stencil, ij_cache_offset)
{
    typename storage_type::meta_data_t meta_(m_d1, m_d2, m_d3);
    storage_type ref(meta_,  0.0, "ref");

    for(int i=m_halo_size; i < m_d1-m_halo_size; ++i)
    {
        for(int j=m_halo_size; j < m_d2-m_halo_size; ++j)
        {
            for(int k=0; k < m_d3; ++k)
            {
                ref(i,j,k) = (m_in(i-1,j,k) + m_in(i+1, j,k) + m_in(i,j-1,k) + m_in(i,j+1,k) ) / (float_type)4.0;
            }
        }
    }

    typedef boost::mpl::vector3<p_in, p_out, p_buff> accessor_list;
    gridtools::domain_type<accessor_list> domain(boost::fusion::make_vector(&m_in, &m_out));

#ifdef __CUDACC__
    gridtools::computation* pstencil =
#else
        boost::shared_ptr<gridtools::computation> pstencil =
#endif
        make_computation<gridtools::BACKEND>
        (
            make_mss // mss_descriptor
            (
                execute<forward>(),
                define_caches(cache<IJ, p_buff, local>()),
                make_esf<functor1>(p_in(), p_buff()), // esf_descriptor
                make_esf<functor2>(p_buff(), p_out()) // esf_descriptor
            ),
            domain, m_grid
        );

    pstencil->ready();

    pstencil->steady();

    pstencil->run();

    pstencil->finalize();

#ifdef __CUDACC__
    m_out.data().update_cpu();
#endif

#ifdef CXX11_ENABLED
    verifier verif(1e-13);
    array<array<uint_t, 2>, 3> halos{{ {m_halo_size,m_halo_size}, {m_halo_size,m_halo_size}, {m_halo_size,m_halo_size} }};
    ASSERT_TRUE(verif.verify(ref, m_out, halos) );
#else
    verifier verif(1e-13, m_halo_size);
    ASSERT_TRUE(verif.verify(ref, m_out));
#endif
}
