#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"
#include "unstructured_grid.hpp"

using namespace gridtools;
using namespace enumtype;

namespace vs_test{

    using backend_t = ::gridtools::backend<Host, icosahedral, Naive >;
    using icosahedral_topology_t = ::gridtools::icosahedral_topology<backend_t>;

    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

    struct test_on_vertexes_functor {
        typedef in_accessor<0, icosahedral_topology_t::vertexes, extent<1> > in;
        typedef inout_accessor<1, icosahedral_topology_t::vertexes> out;
        typedef in_accessor<2, icosahedral_topology_t::vertexes, extent<1> > ipos;
        typedef in_accessor<3, icosahedral_topology_t::vertexes, extent<1> > cpos;
        typedef in_accessor<4, icosahedral_topology_t::vertexes, extent<1> > jpos;
        typedef in_accessor<5, icosahedral_topology_t::vertexes, extent<1> > kpos;
        typedef boost::mpl::vector6<in, out, ipos, cpos, jpos, kpos> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            auto ff = [](const double _in, const double _res) -> double
                {
                return _in+_res;
                 };

            /**
               This interface checks that the location types are compatible with the accessors
             */
            eval(out()) = eval(on_vertexes(ff, 0.0, in()));
        }
    };
}

using namespace vs_test;

TEST(test_stencil_on_vertexes, run) {

    typedef gridtools::layout_map<2,1,0> layout_t;

    using vertex_storage_type = typename backend_t::storage_t<icosahedral_topology_t::vertexes, double>;

    const uint_t halo_nc = 1;
    const uint_t halo_mc = 1;
    const uint_t halo_k = 0;
    const uint_t d3=6+halo_k*2;
    const uint_t d1=6+halo_nc*2;
    const uint_t d2=6+halo_mc*2;
    icosahedral_topology_t icosahedral_grid( d1, d2, d3 );

    auto in_vertexes = icosahedral_grid.make_storage<icosahedral_topology_t::vertexes, double>("in");
    auto i_vertexes = icosahedral_grid.make_storage<icosahedral_topology_t::vertexes, double>("i");
    auto j_vertexes = icosahedral_grid.make_storage<icosahedral_topology_t::vertexes, double>("j");
    auto c_vertexes = icosahedral_grid.make_storage<icosahedral_topology_t::vertexes, double>("c");
    auto k_vertexes = icosahedral_grid.make_storage<icosahedral_topology_t::vertexes, double>("k");
    auto out_vertexes = icosahedral_grid.make_storage<icosahedral_topology_t::vertexes, double>("out");
    auto ref_vertexes = icosahedral_grid.make_storage<icosahedral_topology_t::vertexes, double>("ref");

    for(int i=0; i < d1; ++i)
    {
        for(int c=0; c < icosahedral_topology_t::vertexes::n_colors::value; ++c)
        {
            for(int j=0; j < d2; ++j)
            {
                for(int k=0; k < d3; ++k)
                {
                    in_vertexes(i,c,j,k) = (uint_t)in_vertexes.meta_data().index(array<uint_t,4>
                        {(uint_t)i,(uint_t)c,(uint_t)j,(uint_t)k});
                    i_vertexes(i,c,j,k) = i;
                    c_vertexes(i,c,j,k) = c;
                    j_vertexes(i,c,j,k) = j;
                    k_vertexes(i,c,j,k) = k;
                }
            }
        }
    }
    out_vertexes.initialize(0.0);
    ref_vertexes.initialize(0.0);

    typedef arg<0, vertex_storage_type> p_in_vertexes;
    typedef arg<1, vertex_storage_type> p_out_vertexes;
    typedef arg<2, vertex_storage_type> p_i_vertexes;
    typedef arg<3, vertex_storage_type> p_c_vertexes;
    typedef arg<4, vertex_storage_type> p_j_vertexes;
    typedef arg<5, vertex_storage_type> p_k_vertexes;

    typedef boost::mpl::vector<p_in_vertexes, p_out_vertexes, p_i_vertexes, p_c_vertexes, p_j_vertexes, p_k_vertexes> accessor_list_t;

    gridtools::domain_type<accessor_list_t> domain(boost::fusion::make_vector(&in_vertexes, &out_vertexes, &i_vertexes, &c_vertexes, &j_vertexes, &k_vertexes) );
    array<uint_t,5> di = {halo_nc, halo_nc, halo_nc, d1 - halo_nc -1, d1};
    array<uint_t,5> dj = {halo_mc, halo_mc, halo_mc, d2 - halo_mc -1, d2};

    gridtools::grid<axis, icosahedral_topology_t> grid_(icosahedral_grid, di, dj);
    grid_.value_list[0] = 0;
    grid_.value_list[1] = d3-1;

#ifdef __CUDACC__
        gridtools::stencil* copy =
#else
            std::shared_ptr<gridtools::stencil> copy =
#endif
            gridtools::make_computation<backend_t >
            (
                domain, grid_
                , gridtools::make_mss // mss_descriptor
                (
                    execute<forward>(),
                    gridtools::make_esf<test_on_vertexes_functor, icosahedral_topology_t, icosahedral_topology_t::vertexes>(
                        p_in_vertexes(), p_out_vertexes(), p_i_vertexes(), p_c_vertexes(), p_j_vertexes(), p_k_vertexes() )
                )
            );
    copy->ready();
    copy->steady();
    copy->run();
    copy->finalize();

    unstructured_grid ugrid(d1, d2, d3);
    for(uint_t i=halo_nc; i < d1-halo_nc; ++i)
    {
        for(uint_t c=0; c < icosahedral_topology_t::vertexes::n_colors::value; ++c)
        {
            for(uint_t j=halo_mc; j < d2-halo_mc+1; ++j)
            {
                for(uint_t k=0; k < d3; ++k)
                {
                    auto neighbours = ugrid.neighbours_of<
                            icosahedral_topology_t::vertexes,
                            icosahedral_topology_t::vertexes>({i,c,j,k});
                    for(auto iter = neighbours.begin(); iter != neighbours.end(); ++iter)
                    {
                        ref_vertexes(i,c,j,k) += in_vertexes(*iter);
                    }
                }
            }
        }
    }

    verifier ver(1e-10);

    array<array<uint_t, 2>, 4> halos = {{ {halo_nc, halo_nc},{0,0},{halo_mc, halo_mc},{halo_k, halo_k} }};
    EXPECT_TRUE(ver.verify(grid_, ref_vertexes, out_vertexes, halos));
}
