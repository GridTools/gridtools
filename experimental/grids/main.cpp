using uint_t = unsigned int;
using int_t = int;
#include <iostream>
#include "grid.hpp"
#include "base_storage.hpp"
#include <common/layout_map.h>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/adapted/mpl/detail/size_impl.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <type_traits>
#include "array_addons.hpp"
#include "placeholders.hpp"
#include "make_stencil.hpp"
#include "iterate_domain.hpp"

using gridtools::layout_map;
using gridtools::wrap_pointer;


using trapezoid_2D = gridtools::trapezoid_2D_colored<gridtools::_backend>;

using cell_storage_type = typename trapezoid_2D::cell_storage_t;
using edge_storage_type = typename trapezoid_2D::edge_storage_t;


struct stencil_on_cells {
    typedef accessor<0, trapezoid_2D::cells> out;
    typedef accessor<1, trapezoid_2D::cells> in;
    typedef accessor<2, trapezoid_2D::edges> out_edges_NOT_USED;
    typedef accessor<3, trapezoid_2D::edges> in_edges;

    template <typename GridAccessors>
    void
    operator()(GridAccessors /*const*/& eval/*, region*/) const {
        // std::cout << "i = " << eval.i()
        //           << " j = " << eval.j()
        //           << std::endl;
        auto ff = [](const double _in, const double _res) -> double
            {
#ifdef _MAIN_CPP_DEBUG_
                std::cout << "#";
#endif
                return _in+_res;
             };


        /**
           This interface checks that the location types are compatible with the accessors
         */
        eval(out()) = eval(reduce_on_cells(ff, 0.0, in())) + eval(reduce_on_edges(ff, 0.0, in_edges()));
    }
};


#define _MAIN_CPP_DEBUG_

struct nested_stencil {
    typedef accessor<0, trapezoid_2D::cells> out;
    typedef accessor<1, trapezoid_2D::cells> in;
    typedef accessor<2, trapezoid_2D::edges> edges0;
    typedef accessor<3, trapezoid_2D::edges> edges1;

    template <typename GridAccessors>
    void
    operator()(GridAccessors /*const*/& eval/*, region*/) const {
        // std::cout << "i = " << eval.i()
        //           << " j = " << eval.j()
        //           << std::endl;
        auto ff = [](const double _in, const double _res) -> double
            {
#ifdef _MAIN_CPP_DEBUG_
                std::cout << "#";
#endif
                return _in+_res+1;
            };
        auto gg = [](const double _in, const double _res) -> double
            {
#ifdef _MAIN_CPP_DEBUG_
                std::cout << "m";
#endif
                return _in+_res+2;
            };
        auto reduction = [](const double _in, const double _res) -> double
            {
#ifdef _MAIN_CPP_DEBUG_
                std::cout << "r";
#endif
                return _in+_res+3;
            };

        auto x = eval(reduce_on_edges(reduction, 0.0, map(gg, edges0(), reduce_on_cells(ff, 0.0, map(identity<double>(), in())))));
        auto y = eval(reduce_on_edges(reduction, 0.0, map(gg, edges0(), reduce_on_cells(ff, 0.0, in()))));
        assert(x==y);
        std:: cout << x << " == " << y << std::endl;
        //eval(out()) = eval(reduce_on_edges(reduction, 0.0, edges0::reduce_on_cells(gg, in()), edges1()));
    }
};

struct stencil_on_edges {
    typedef accessor<0, trapezoid_2D::cells> out_NOT_USED;
    typedef accessor<1, trapezoid_2D::cells> in;
    typedef accessor<2, trapezoid_2D::edges> out_edges;
    typedef accessor<3, trapezoid_2D::edges> in_edges;

    template <typename GridAccessors>
    void
    operator()(GridAccessors /*const*/& eval/*, region*/) const {
        // std::cout << "i = " << eval.i()
        //           << " j = " << eval.j()
        //           << std::endl;
        auto ff = [](const double _in, const double _res) -> double
            {
#ifdef _MAIN_CPP_DEBUG_
                std::cout << "e";
#endif
                return _in+_res;
            };
        eval(out_edges()) = eval(reduce_on_cells(ff, 0.0, in()))+ eval(reduce_on_edges(ff, 0.0, in_edges()));
    }
};

#define EVAL(f,x,y)                                                     \
    std::cout << #f << ": " << gridtools::array<decltype(x),2>{x,y} << " -> " << (grid.f({x,y})) << std::endl

#define EVAL_C(f,x,y, result)                                                    \
    std::cout << #f << ": " << gridtools::array<decltype(x),2>{x,y} << " -> " << (grid.f({x,y})) << ", expect " << result; \
    std::cout << ": Passed? " << std::boolalpha << (grid.f({x,y}) == result) << std::endl

#define _EVAL_C(l1,l2,x,y, result)                                        \
    std::cout << #l1 << "->" << #l2 << ": " << gridtools::array<decltype(x),2>{x,y} << " -> " << (grid.neighbors({x,y}, trapezoid_2D::l1(), trapezoid_2D::l2())) << ", expect " << result; \
    std::cout << ": Passed? " << std::boolalpha << (grid.neighbors({x,y},trapezoid_2D::l1(),trapezoid_2D::l2()) == result) << std::endl

#define _EVAL_I(l1,l2,x,y)                                              \
    std::cout << #l1 << "->" << #l2 << ": " << gridtools::array<decltype(x),2>{x,y} << " -> " << (grid.neighbors_indices({x,y}, trapezoid_2D::l1(), trapezoid_2D::l2())) << std::endl;

template <typename Array>
uint_t product(Array && x) {
    uint_t r = 1;
    for (int i = 0; i < x.size(); ++i) {
        r *= x[i];
    }
    return r;
};

int main() {
    uint_t NC = trapezoid_2D::u_cell_size_i(6);
    uint_t MC = trapezoid_2D::u_cell_size_j(12);

    uint_t NE = trapezoid_2D::u_edge_size_i(6);
    uint_t ME = trapezoid_2D::u_edge_size_j(12);

    std::cout << "NC = " << NC << " "
              << "MC = " << MC
              << std::endl;

    std::cout << "NE = " << NE << " "
              << "ME = " << ME
              << std::endl;

    uint_t d3=2;
    trapezoid_2D grid( 6, 12, d3 );

    cell_storage_type cells(grid.size(gridtools::location_type<0>()));
    edge_storage_type edges(grid.size(gridtools::location_type<1>()));

    EVAL_C(cell2cells_ll_p0, 1, 1, (gridtools::array<uint_t,3>{9*d3, 24*d3, 25*d3}));
    EVAL_C(cell2cells_ll_p0/*color0*/, 1, 2,/*coords*/ (gridtools::array<uint_t,3>{/*offsets*/10*d3, 25*d3, 26*d3}));
    EVAL_C(cell2cells_ll_p1/*color1*/, 1, 3, (gridtools::array<uint_t,3>{19*d3, 20*d3, 35*d3}));
    EVAL_C(cell2cells_ll_p1, 1, 4, (gridtools::array<uint_t,3>{20*d3, 21*d3, 36*d3}));
    _EVAL_C(cells,cells, 2, 3, (gridtools::array<uint_t,3>{33*d3, 34*d3, 49*d3}));
    _EVAL_C(cells,cells, 2, 4, (gridtools::array<uint_t,3>{26*d3, 41*d3, 42*d3}));
    _EVAL_C(cells,cells, 3, 3, (gridtools::array<uint_t,3>{49*d3, 50*d3, 65*d3}));
    _EVAL_C(cells,cells, 3, 4, (gridtools::array<uint_t,3>{42*d3, 57*d3, 58*d3}));

    EVAL_C(edge2edges_ll_p0, 2, 3, (gridtools::array<uint_t,4>{66*d3,67*d3,59*d3,82*d3}));
    EVAL_C(edge2edges_ll_p1, 2, 3, (gridtools::array<uint_t,4>{43*d3,28*d3,51*d3,67*d3}));
    EVAL_C(edge2edges_ll_p2, 2, 3, (gridtools::array<uint_t,4>{51*d3,59*d3,52*d3,83*d3}));
    _EVAL_C(edges,edges, 2, 2, (gridtools::array<uint_t,4>{48*d3,56*d3,49*d3,80*d3}));
    _EVAL_C(edges,edges, 2, 3, (gridtools::array<uint_t,4>{64*d3,57*d3,65*d3,80*d3}));
    _EVAL_C(edges,edges, 2, 4, (gridtools::array<uint_t,4>{41*d3,26*d3,49*d3,65*d3}));

    EVAL_C(cell2edges_ll_p0, 2, 3, (gridtools::array<uint_t,3>{51*d3,59*d3,67*d3}));
    EVAL_C(cell2edges_ll_p1, 2, 3, (gridtools::array<uint_t,3>{67*d3,52*d3,83*d3}));
    _EVAL_C(cells,edges, 2, 3, (gridtools::array<uint_t,3> {65*d3,50*d3,81*d3}));
    _EVAL_C(cells,edges, 2, 4, (gridtools::array<uint_t,3> {58*d3,50*d3,66*d3}));

    EVAL_C(edge2cells_ll_p0, 2, 3, (gridtools::array<uint_t,2>{42*d3,35*d3}));
    EVAL_C(edge2cells_ll_p1, 2, 3, (gridtools::array<uint_t,2>{27*d3,35*d3}));
    EVAL_C(edge2cells_ll_p2, 2, 3, (gridtools::array<uint_t,2>{35*d3,43*d3}));
    _EVAL_C(edges,cells, 2, 3, (gridtools::array<uint_t,2>    {33*d3,40*d3}));
    _EVAL_C(edges,cells, 2, 4, (gridtools::array<uint_t,2>    {25*d3,33*d3}));
    _EVAL_C(edges,cells, 2, 5, (gridtools::array<uint_t,2>    {33*d3,41*d3}));




    _EVAL_I(cells,cells, 2, 3);
    _EVAL_I(cells,cells, 2, 4);
    _EVAL_I(cells,cells, 3, 3);
    _EVAL_I(cells,cells, 3, 4);

    _EVAL_I(edges,edges, 2, 2);
    _EVAL_I(edges,edges, 2, 3);
    _EVAL_I(edges,edges, 2, 4);

    _EVAL_I(cells,edges, 2, 3);
    _EVAL_I(cells,edges, 2, 4);

    _EVAL_I(edges,cells, 2, 3);
    _EVAL_I(edges,cells, 2, 4);
    _EVAL_I(edges,cells, 2, 5);


    cell_storage_type cells_out(grid.size(gridtools::location_type<0>()));
    edge_storage_type edges_out(grid.size(gridtools::location_type<1>()));
    // cell_storage_type cells_out(product(trapezoid_2D::u_cell_size(NC, MC)));
    // edge_storage_type edges_out(product(trapezoid_2D::u_edge_size(NE, ME)));



    typedef arg<0, trapezoid_2D::cells> out_cells;
    typedef arg<1, trapezoid_2D::cells> in_cells;
    typedef arg<2, trapezoid_2D::edges> out_edges;
    typedef arg<3, trapezoid_2D::edges> in_edges;


    std::cout << "#############################################################################################################################################################" << std::endl;
    std::cout << "#############################################################################################################################################################" << std::endl;
    std::cout << "CASE # 1" << std::endl;
    std::cout << "#############################################################################################################################################################" << std::endl;
    std::cout << "#############################################################################################################################################################" << std::endl;

    {
        auto x = make_esf<stencil_on_cells, trapezoid_2D, trapezoid_2D::cells>
            (out_cells(), in_cells(), out_edges(), in_edges());


        // gridtools::domain_type<boost::mpl::vector<in_cells, out_cells, out_edges, in_edges> >
        //     (&cells_out, &cells, &edges_out, &edges);

        auto ptrs = boost::fusion::vector<cell_storage_type*,
                                          cell_storage_type*,
                                          edge_storage_type*,
                                          edge_storage_type*>
            (&cells_out, &cells, &edges_out, &edges);

        iterate_domain<boost::mpl::vector<in_cells, out_cells, out_edges, in_edges>,
                      trapezoid_2D, trapezoid_2D::cells> acc
            (ptrs, grid);


        struct _coords {
            int_t lb0, ub0;
            int_t lb1, ub1;

            _coords(int lb0, int ub0, int lb1, int ub1)
                : lb0(lb0)
                , ub0(ub0)
                , lb1(lb1)
                , ub1(ub1)
            {}
        } coords(1, NC-1-1, 2, MC-2-1); // closed intervals

        gridtools::colored_backend::run(acc, x, coords);

    }

    std::cout << "#############################################################################################################################################################" << std::endl;
    std::cout << "#############################################################################################################################################################" << std::endl;
    std::cout << "CASE # 2" << std::endl;
    std::cout << "#############################################################################################################################################################" << std::endl;
    std::cout << "#############################################################################################################################################################" << std::endl;

    {
        auto x = make_esf<stencil_on_edges, trapezoid_2D, trapezoid_2D::edges>
            (out_cells(), out_cells(), out_edges(), in_edges());

        iterate_domain<boost::mpl::vector<in_cells, out_cells, out_edges, in_edges>,
                      trapezoid_2D, trapezoid_2D::cells> acc
            (boost::fusion::vector<cell_storage_type*, cell_storage_type*, edge_storage_type*, edge_storage_type*>
             (&cells_out, &cells, &edges_out, &edges), grid);


        struct _coords {
            int_t lb0, ub0;
            int_t lb1, ub1;

            _coords(int lb0, int ub0, int lb1, int ub1)
                : lb0(lb0)
                , ub0(ub0)
                , lb1(lb1)
                , ub1(ub1)
            {}
        } coords(1, NE-1-1, 3, ME-3);

        gridtools::colored_backend::run(acc, x, coords);

    }


    std::cout << "#############################################################################################################################################################" << std::endl;
    std::cout << "#############################################################################################################################################################" << std::endl;
    std::cout << "CASE # 3" << std::endl;
    std::cout << "#############################################################################################################################################################" << std::endl;
    std::cout << "#############################################################################################################################################################" << std::endl;

    {
        auto x = make_esf<nested_stencil, trapezoid_2D, trapezoid_2D::cells>
            (out_cells(), in_cells(), out_edges(), in_edges());

        iterate_domain<boost::mpl::vector<in_cells, out_cells, out_edges, in_edges>,
                      trapezoid_2D, trapezoid_2D::cells> acc
            (boost::fusion::vector<cell_storage_type*, cell_storage_type*, edge_storage_type*, edge_storage_type*>
             (&cells_out, &cells, &edges_out, &edges), grid);


        struct _coords {
            int_t lb0, ub0;
            int_t lb1, ub1;

            _coords(int lb0, int ub0, int lb1, int ub1)
                : lb0(lb0)
                , ub0(ub0)
                , lb1(lb1)
                , ub1(ub1)
            {}
        } coords(1, NC-1-1, 2, MC-2-1);

        gridtools::colored_backend::run(acc, x, coords);
    }

    std::cout << std::endl;
    return 0;
}
