#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"
#include "unstructured_grid.hpp"

using namespace gridtools;
using namespace enumtype;

namespace nested_test{

    using backend_t = ::gridtools::backend<Host, Naive >;
    using icosahedral_topology_t = ::gridtools::icosahedral_topology<backend_t>;

    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

    struct nested_stencil {
        typedef in_accessor<0, icosahedral_topology_t::cells, radius<2>> in_cells;
        typedef in_accessor<1, icosahedral_topology_t::edges, radius<1>> in_edges;
        typedef in_accessor<2, icosahedral_topology_t::edges, radius<1> > ipos;
        typedef in_accessor<3, icosahedral_topology_t::edges, radius<1> > cpos;
        typedef in_accessor<4, icosahedral_topology_t::edges, radius<1> > jpos;
        typedef in_accessor<5, icosahedral_topology_t::edges, radius<1> > kpos;
        typedef inout_accessor<6, icosahedral_topology_t::edges> out_edges;

        typedef boost::mpl::vector<in_cells, in_edges, ipos, cpos, jpos, kpos, out_edges> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            auto ff = [](const double _in, const double _res) -> double
                {
                std::cout << "INNER FF " << _in << " " << _res << " " << _in+_res+1 << std::endl;

                    return _in+_res+1;
                };
            auto gg = [](const double _in, const double _res) -> double
                {
                std::cout << "MAP ON EDGES " << _in << " " << _res << " " << _in+_res+2 << std::endl;

                    return _in+_res+2;
                };
            auto reduction = [](const double _in, const double _res) -> double
                {
                std::cout << "RED ON EDGES " << _in << " " << _res << " " << _in+_res+3 << std::endl;
                    return _in+_res+3;
                };

            std::cout << "FOR i,c,j,k " << eval(ipos()) << " " << eval(jpos()) << " " <<
                         eval(cpos()) << " " << eval(kpos()) << std::endl;

//            auto x = eval(on_edges(reduction, 0.0,
//                map(gg, in_edges(), on_cells(ff, 0.0, map(identity<double>(), in_cells())))));
            auto y = eval(on_edges(reduction, 0.0,
                map(gg, in_edges(),
                    on_cells(ff, 0.0, in_cells())
                    )
                                   )
                          );
            //eval(out()) = eval(reduce_on_edges(reduction, 0.0, edges0::reduce_on_cells(gg, in()), edges1()));
        }
    };

}

using namespace nested_test;

TEST(test_stencil_nested_on, run) {

    typedef gridtools::layout_map<2,1,0> layout_t;

    using cell_storage_type = typename backend_t::storage_t<icosahedral_topology_t::cells, double>;
    using edge_storage_type = typename backend_t::storage_t<icosahedral_topology_t::edges, double>;

    const uint_t halo_nc = 1;
    const uint_t halo_mc = 1;
    const uint_t halo_k = 0;
    const uint_t d3=6+halo_k*2;
    const uint_t d1=6+halo_nc*2;
    const uint_t d2=6+halo_mc*2;
    icosahedral_topology_t icosahedral_grid( d1, d2, d3 );

    cell_storage_type in_cells = icosahedral_grid.make_storage<icosahedral_topology_t::cells, double>("in_cell");
    edge_storage_type in_edges = icosahedral_grid.make_storage<icosahedral_topology_t::edges, double>("in_edge");
    edge_storage_type out_edges = icosahedral_grid.make_storage<icosahedral_topology_t::edges, double>("out_edge");

    edge_storage_type i_edges = icosahedral_grid.make_storage<icosahedral_topology_t::edges, double>("i");
    edge_storage_type j_edges = icosahedral_grid.make_storage<icosahedral_topology_t::edges, double>("j");
    edge_storage_type c_edges = icosahedral_grid.make_storage<icosahedral_topology_t::edges, double>("c");
    edge_storage_type k_edges = icosahedral_grid.make_storage<icosahedral_topology_t::edges, double>("k");
    edge_storage_type ref_edges = icosahedral_grid.make_storage<icosahedral_topology_t::edges, double>("ref");

    for(int i=0; i < d1; ++i)
    {
        for(int c=0; c < icosahedral_topology_t::edges::n_colors::value; ++c)
        {
            for(int j=0; j < d2; ++j)
            {
                for(int k=0; k < d3; ++k)
                {
                    if(c < icosahedral_topology_t::cells::n_colors::value)
                    {
                        in_cells(i,c,j,k) = in_cells.meta_data().index(array<uint_t,4>
                            {(uint_t)i,(uint_t)c,(uint_t)j,(uint_t)k});
                    }
                    in_edges(i,c,j,k) = in_edges.meta_data().index(array<uint_t,4>
                        {(uint_t)i,(uint_t)c,(uint_t)j,(uint_t)k});

                    i_edges(i,c,j,k) = i;
                    c_edges(i,c,j,k) = c;
                    j_edges(i,c,j,k) = j;
                    k_edges(i,c,j,k) = k;
                }
            }
        }
    }
    ref_edges.initialize(0.0);

    typedef arg<0, cell_storage_type> p_in_cells;
    typedef arg<1, edge_storage_type> p_in_edges;
    typedef arg<2, edge_storage_type> p_i_edges;
    typedef arg<3, edge_storage_type> p_c_edges;
    typedef arg<4, edge_storage_type> p_j_edges;
    typedef arg<5, edge_storage_type> p_k_edges;
    typedef arg<6, edge_storage_type> p_out_edges;

    typedef boost::mpl::vector<
        p_in_cells, p_in_edges, p_i_edges, p_c_edges, p_j_edges, p_k_edges, p_out_edges
    > accessor_list_t;

    gridtools::domain_type<accessor_list_t> domain(boost::fusion::make_vector
        (&in_cells, &in_edges, &i_edges, &c_edges, &j_edges, &k_edges, &out_edges) );
    array<uint_t,5> di = {halo_nc, halo_nc, halo_nc, d1 - halo_nc -1, d1};
    array<uint_t,5> dj = {halo_mc, halo_mc, halo_mc, d2 - halo_mc -1, d2};

    gridtools::grid<axis, icosahedral_topology_t> grid_(icosahedral_grid, di, dj);
    grid_.value_list[0] = 0;
    grid_.value_list[1] = d3-1;

#ifdef __CUDACC__
        gridtools::computation* copy =
#else
            boost::shared_ptr<gridtools::computation> copy =
#endif
            gridtools::make_computation<backend_t >
            (
                domain, grid_
                , gridtools::make_mss // mss_descriptor
                (
                    execute<forward>(),
                    gridtools::make_esf<nested_stencil, icosahedral_topology_t, icosahedral_topology_t::cells>
                        (p_in_cells(), p_in_edges(), p_i_edges(), p_c_edges(),
                         p_j_edges(), p_k_edges(), p_out_edges() )
                )
            );
    copy->ready();
    copy->steady();
    copy->run();

    unstructured_grid ugrid(d1, d2, d3);
    for(uint_t i=halo_nc; i < d1-halo_nc; ++i)
    {
        for(uint_t c=0; c < icosahedral_topology_t::edges::n_colors::value; ++c)
        {
            for(uint_t j=halo_mc; j < d2-halo_mc; ++j)
            {
                for(uint_t k=0; k < d3; ++k)
                {
                    std::cout << "REFFOR i,c,j,k " << i_edges(i,c,j,k) << " " << c_edges(i,c,j,k) << " " <<
                        j_edges(i,c,j,k) << " " << k_edges(i,c,j,k) << std::endl;

                    double acc=0.0;
                    auto neighbours = ugrid.neighbours_of<
                            icosahedral_topology_t::edges,
                            icosahedral_topology_t::edges>({i,c,j,k});
                    for(auto edge_iter = neighbours.begin(); edge_iter != neighbours.end(); ++edge_iter)
                    {
                        auto innercell_neighbours = ugrid.neighbours_of<
                                icosahedral_topology_t::edges,
                                icosahedral_topology_t::cells>(*edge_iter);
                        for(auto cell_iter = innercell_neighbours.begin(); cell_iter != innercell_neighbours.end();
                            ++cell_iter)
                        {
                            std::cout << "REF INNER FF " << in_cells(*cell_iter) << " " << acc << " " <<
                                     acc +in_cells(*cell_iter) +1 << std::endl;

                            acc += in_cells(*cell_iter)+1;
                        }
                        std::cout << "MAP ON EDGES " << acc << " " << in_edges(*edge_iter) << " " <<
                               (acc+in_edges(*edge_iter)+2) << std::endl;

                    }

                    std::cout << "RED ON EDGES " << (acc+in_edges(i,c,j,k)+2) << " " <<
                              ref_edges(i,c,j,k) << " " << (acc+in_edges(i,c,j,k)+2)+3 << std::endl;

                    ref_edges(i,c,j,k) += (acc+in_edges(i,c,j,k)+2)+3;
                }
            }
        }
    }

    verifier ver(1e-10);

    array<array<uint_t, 2>, 4> halos = {{ {halo_nc, halo_nc},{0,0},{halo_mc, halo_mc},{halo_k, halo_k} }};
    EXPECT_TRUE(ver.verify(grid_, ref_edges, out_edges, halos));
}
