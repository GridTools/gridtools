#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>

using namespace gridtools;
using namespace enumtype;

typedef interval< level< 0, -1 >, level< 1, -1 > > x_interval;
typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

struct TensionShearFunction
{
    using T_sqr_s  = inout_accessor<0>;
    using S_sqr_uv = inout_accessor<1>;

    using u_in = in_accessor<2, extent<-1, 0,  0, 1>>;
    using v_in = in_accessor<3, extent< 0, 1, -1, 0>>;

    using arg_list = boost::mpl::vector<T_sqr_s, S_sqr_uv, u_in, v_in>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(const Evaluation& eval, x_interval)
    { }
};

/**
 * @brief Function computing the coefficients for the Smagorinsky diffusion
 *
 * Flops: 8 Nadd + 5 Nmul + 4 Ncmp + 2 Nsqrt= 27
 *
 * Assuming sqrt costs approximately 5 flops.
 *
 * Refrence:
 *  - STELLA: dycore/HorizontalDiffusionSmagorinsky.cpp
 */
struct SmagCoeffFunction
{
    using smag_u = inout_accessor<0>;
    using smag_v = inout_accessor<1>;

    using T_sqr_s  = in_accessor<2, extent<0, 1, 0, 1>>;
    using S_sqr_uv = in_accessor<3, extent<-1, 0, -1, 0>>;

    using arg_list = boost::mpl::vector<smag_u,
                                        smag_v,
                                        T_sqr_s,
                                        S_sqr_uv>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(const Evaluation& eval, x_interval)
    {  }
};

/**
 * @brief Function updating the horizontal velocities using the Smagorinsy coefficients
 *
 * Flops: 14 Nadd + 8 Nmul = 22
 *
 * Refrence:
 *  - STELLA: dycore/HorizontalDiffusionSmagorinsky.cpp
 */
struct SmagUpdateFunction
{
    using u_out = inout_accessor<0>;
    using v_out = inout_accessor<1>;

    using u_in  = in_accessor<2, extent<-1, 1, -1, 1>>;
    using v_in = in_accessor<3, extent<-1, 1, -1, 1>>;
    using smag_u = in_accessor<4>;
    using smag_v = in_accessor<5>;

    using arg_list = boost::mpl::vector<u_out,
                                        v_out,
                                        u_in,
                                        v_in,
                                        smag_u,
                                        smag_v>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(const Evaluation& eval, x_interval)
    { }
};

#ifdef __CUDACC__
#define BACKEND backend<enumtype::Cuda,GRIDBACKEND, enumtype::Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<enumtype::Host,GRIDBACKEND, enumtype::Block >
#else
#define BACKEND backend<enumtype::Host,GRIDBACKEND, enumtype::Naive >
#endif
#endif


TEST(multiple_outputs, compute_extents) {
    typedef layout_map< 2, 1, 0 > layout_t;
    typedef BACKEND::storage_info< 0, layout_t > storage_info_type;
    typedef BACKEND::storage_type< float_type, storage_info_type >::type storage_type;
    typedef BACKEND::temporary_storage_type< float_type, storage_info_type >::type tmp_storage_type;

    storage_info_type meta_data_(10, 10, 10);
    storage_type dummy(meta_data_, 0., "dummy");

    using T_sqr_s  = arg<0, tmp_storage_type>;
    using S_sqr_uv = arg<1, tmp_storage_type>;
    using smag_u   = arg<2, tmp_storage_type>;
    using smag_v   = arg<3, tmp_storage_type>;

    // Output fields
    using u_out = arg<4, storage_type>;
    using v_out = arg<5, storage_type>;

    // Input fields
    using u_in = arg<6, storage_type>;
    using v_in = arg<7, storage_type>;

    using arg_list = boost::mpl::vector<
        // Temporaries
        T_sqr_s,
        S_sqr_uv,
        smag_u,
        smag_v,

        // Output fields
        u_out,
        v_out,

        // Input fields
        u_in,
        v_in>;

    domain_type< arg_list > domain(boost::fusion::make_vector(&dummy, &dummy, &dummy, &dummy));

    uint_t di[5] = {2, 2, 0, 7, 10};
    uint_t dj[5] = {2, 2, 0, 7, 10};
    grid<axis> grid_(di,dj);

    grid_.value_list[0] = 0;
    grid_.value_list[1] = 9;


    auto computation = make_computation<BACKEND>
        (
         domain, grid_,
         make_mss
         (
          execute<forward>(),
          make_esf<TensionShearFunction>(T_sqr_s(),
                                         S_sqr_uv(),
                                         u_in(),
                                         v_in()),
          make_esf<SmagCoeffFunction>(smag_u(),
                                      smag_v(),
                                      T_sqr_s(),
                                      S_sqr_uv()),
          make_esf<SmagUpdateFunction>(u_out(),
                                       v_out(),
                                       u_in(),
                                       v_in(),
                                       smag_u(),
                                       smag_v())
          )
         );

    EXPECT_TRUE(true);
}
