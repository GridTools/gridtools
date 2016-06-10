#include <boost/mpl/equal.hpp>
#include <boost/shared_ptr.hpp>

#include "gtest/gtest.h"

#include "common/defs.hpp"
#include "stencil_composition/stencil_composition.hpp"
#include "stencil_composition/make_computation.hpp"
#include "tools/verifier.hpp"

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
    typedef accessor< 0, enumtype::in, extent< -1, 1, -1, 1 > > in;
    typedef accessor< 1, enumtype::inout > out;
    typedef boost::mpl::vector< in, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
        eval(out()) =
            (eval(in(-1, 0, 0)) + eval(in(1, 0, 0)) + eval(in(0, -1, 0)) + eval(in(0, 1, 0))) / (float_type)4.0;
    }
};

struct functor3 {
    typedef accessor< 0, enumtype::in > in;
    typedef accessor< 1, enumtype::inout > out;
    typedef boost::mpl::vector< in, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
        eval(out()) = eval(in()) + 1;
    }
};

#ifdef __CUDACC__
#define BACKEND backend< Cuda, structured, Block >
#else
#define BACKEND backend< Host, structured, Block >
#endif

typedef layout_map<2,1,0> layout_ijk_t;
typedef gridtools::BACKEND::storage_type< float_type, gridtools::BACKEND::storage_info< 0, layout_ijk_t > >::type
    storage_type;
typedef gridtools::BACKEND::temporary_storage_type< float_type,
    gridtools::BACKEND::storage_info< 0, layout_ijk_t > >::type tmp_storage_type;

typedef arg<0, storage_type> p_in;
typedef arg<1, storage_type> p_out;
typedef arg<2, tmp_storage_type> p_buff;
typedef arg< 3, tmp_storage_type > p_buff_2;
typedef arg< 4, tmp_storage_type > p_buff_3;
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
    typename storage_type::storage_info_type m_meta;
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
        m_in(m_meta, 0., "in"),
        m_out(m_meta, 0., "out")
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
    SetUp();
    typedef boost::mpl::vector3<p_in, p_out, p_buff> accessor_list;
    gridtools::aggregator_type<accessor_list> domain(boost::fusion::make_vector(&m_in, &m_out));

#ifdef CXX11_ENABLED
    auto
#else
#ifdef __CUDACC__
    gridtools::stencil *
#else
    boost::shared_ptr< gridtools::stencil >
#endif
#endif
        pstencil = make_computation< gridtools::BACKEND >
        (domain,
         m_grid,
         make_multistage // mss_descriptor
         (execute< forward >(),
          define_caches(cache< IJ, local >(p_buff())),
          make_stage< functor1 >(p_in(), p_buff()),
          make_stage< functor1 >(p_buff(), p_out())));

    pstencil->ready();

    pstencil->steady();

    pstencil->run();

    pstencil->finalize();

#ifdef __CUDACC__
    m_out.d2h_update();
#endif
#ifdef CXX11_ENABLED
#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-12);
#endif
    array<array<uint_t, 2>, 3> halos{{ {m_halo_size,m_halo_size}, {m_halo_size,m_halo_size}, {m_halo_size,m_halo_size} }};
    ASSERT_TRUE(verif.verify(m_grid, m_in, m_out, halos) );
#else
#if FLOAT_PRECISION == 4
    verifier verif(1e-6, m_halo_size);
#else
    verifier verif(1e-12, m_halo_size);
#endif
    ASSERT_TRUE(verif.verify(m_grid, m_in, m_out) );
#endif
}

TEST_F(cache_stencil, ij_cache_offset)
{
    SetUp();
    typename storage_type::storage_info_type meta_(m_d1, m_d2, m_d3);
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
    gridtools::aggregator_type<accessor_list> domain(boost::fusion::make_vector(&m_in, &m_out));

#ifdef CXX11_ENABLED
    auto
#else
#ifdef __CUDACC__
    gridtools::stencil *
#else
    boost::shared_ptr< gridtools::stencil >
#endif
#endif
        pstencil = make_computation< gridtools::BACKEND >(domain,
            m_grid,
            make_multistage // mss_descriptor
            (execute< forward >(),
                                                              define_caches(cache< IJ, local >(p_buff())),
                                                              make_stage< functor1 >(p_in(), p_buff()), // esf_descriptor
                                                              make_stage< functor2 >(p_buff(), p_out()) // esf_descriptor
                                                              ));

    pstencil->ready();

    pstencil->steady();

    pstencil->run();

    pstencil->finalize();

#ifdef __CUDACC__
    m_out.d2h_update();
#endif

#ifdef CXX11_ENABLED
#if FLOAT_PRECISION == 4
    verifier verif(1e-6);
#else
    verifier verif(1e-12);
#endif
    array<array<uint_t, 2>, 3> halos{{ {m_halo_size,m_halo_size}, {m_halo_size,m_halo_size}, {m_halo_size,m_halo_size} }};
    ASSERT_TRUE(verif.verify(m_grid, ref, m_out, halos) );
#else
#if FLOAT_PRECISION == 4
    verifier verif(1e-6, m_halo_size);
#else
    verifier verif(1e-12, m_halo_size);
#endif
    ASSERT_TRUE(verif.verify(m_grid, ref, m_out));
#endif
}

TEST_F(cache_stencil, multi_cache) {
    SetUp();
    typename storage_type::storage_info_type meta_(m_d1, m_d2, m_d3);
    storage_type ref(meta_, 0.0, "ref");

    for (int i = m_halo_size; i < m_d1 - m_halo_size; ++i) {
        for (int j = m_halo_size; j < m_d2 - m_halo_size; ++j) {
            for (int k = 0; k < m_d3; ++k) {
                ref(i, j, k) = (m_in(i, j, k) + 4);
            }
        }
    }

    typedef boost::mpl::vector5< p_in, p_out, p_buff, p_buff_2, p_buff_3 > accessor_list;
    gridtools::aggregator_type< accessor_list > domain(boost::fusion::make_vector(&m_in, &m_out));

#ifdef CXX11_ENABLED
    auto
#else
#ifdef __CUDACC__
    gridtools::stencil *
#else
    boost::shared_ptr< gridtools::stencil >
#endif
#endif
        stencil = make_computation< gridtools::BACKEND >(
            domain,
            m_grid,
            make_multistage // mss_descriptor
            (execute< forward >(),
                // test if define_caches works properly with multiple vectors of caches.
                // in this toy example two vectors are passed (IJ cache vector for p_buff
                // and p_buff_2, IJ cache vector for p_buff_3)
                define_caches(cache< IJ, local >(p_buff(), p_buff_2()), cache< IJ, local >(p_buff_3())),
                make_stage< functor3 >(p_in(), p_buff()),       // esf_descriptor
                make_stage< functor3 >(p_buff(), p_buff_2()),   // esf_descriptor
                make_stage< functor3 >(p_buff_2(), p_buff_3()), // esf_descriptor
                make_stage< functor3 >(p_buff_3(), p_out())     // esf_descriptor
                ));
    stencil->ready();

    stencil->steady();

    stencil->run();

    stencil->finalize();

#ifdef __CUDACC__
    m_out.d2h_update();
#endif

#ifdef CXX11_ENABLED
    verifier verif(1e-13);
    array< array< uint_t, 2 >, 3 > halos{
        {{m_halo_size, m_halo_size}, {m_halo_size, m_halo_size}, {m_halo_size, m_halo_size}}};
    ASSERT_TRUE(verif.verify(m_grid, ref, m_out, halos));
#else
    verifier verif(1e-13, m_halo_size);
    ASSERT_TRUE(verif.verify(m_grid, ref, m_out));
#endif
}
