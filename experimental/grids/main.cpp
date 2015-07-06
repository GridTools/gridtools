using uint_t = unsigned int;
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
#include "accessor.hpp"



using gridtools::layout_map;
using gridtools::wrap_pointer;


using trapezoid_2D = gridtools::trapezoid_2D_no_tile<gridtools::_backend>;

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
            {std::cout << "#"; return _in+_res;};

        /**
           Interface that do not check if the location types are correct
         */
        eval(out()) = eval(on_neighbors(in(), ff, 0.0)) + eval(on_neighbors(in_edges(), ff, 0.0));

        /**
           This interface checks that the location types are compatible with the accessors
         */
        eval(out()) = eval(on_cells(in(), ff, 0.0)) + eval(on_edges(in_edges(),ff, 0.0));

        /**
           This is the most concise interface but maybe not intuitive
         */
        eval(out()) = eval(in()(ff, 0.0)) + eval(in_edges()(ff, 0.0));

        /**
           This interface cannot mistake the location types, since they are incoded in the accessor types ones
         */
        eval(out()) = eval(in::neighbors(ff, 0.0)) + eval(in_edges::neighbors(ff, 0.0));
    }
};

struct stencil_on_edges_cells {
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
        auto nested_reduction = [](const double _in, const double _res) -> double
            {std::cout << "#"; return _in+_res;};
        auto map = [](const double _in, const double _from_neighbors) -> double
            {std::cout << "."; return _in+_from_neighbors;};
        auto top_reduction = [](const double _in, const double _res) -> double
            {std::cout << "+"; return _in+_res;};

        /**
           Interface that do not check if the location types are correct
         */
        eval(out()) = eval(on_neighbors(in_edges(), map, on_neighbors(in(), nested_reduction, 0.0), top_reduction, 0.0 ));

        /**
           This interface checks that the location types are compatible with the accessors
         */
        eval(out()) = eval(on_edges(in_edges(), map, on_cells(in(), nested_reduction, 0.0), top_reduction, 0.0));

        /**
           This interface cannot mistake the location types, since they are incoded in the accessor types ones
         */
        eval(out()) = eval(in_edges::neighbors(map, in::neighbors(nested_reduction, 0.0), top_reduction, 0.0));

        /**
           You can mix interfaces!
         */
        eval(out()) = eval(in_edges()(map, in::neighbors(nested_reduction, 0.0), top_reduction, 0.0));

    }
};

struct stencil_on_cells_edges {
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
            {std::cout << "#"; return _in+_res;};
        auto gg = [](const double _in, const double _res) -> double
            {std::cout << "m"; return _in+_res;};
        auto reduction = [](const double _in, const double _res) -> double
            {std::cout << "r"; return _in+_res;};

        eval(out()) = eval(on_cells(in(), gg, on_edges(in_edges(), ff, 0.0), reduction, 0.0));
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
        auto ff = [](const double _in, const double _res) -> double {std::cout << "e"; return _in+_res;};
        eval(out_edges()) = eval(on_neighbors(in(), ff, 0.0))+ eval(on_neighbors(in_edges(), ff, 0.0));
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

    cell_storage_type cells(product(trapezoid_2D::u_cell_size(gridtools::array<uint_t, 2>{NC, MC})));
    edge_storage_type edges(product(trapezoid_2D::u_edge_size(gridtools::array<uint_t, 2>{NE, ME})));

    trapezoid_2D grid(/*cells, edges,*/ 6, 12);

    EVAL_C(cell2cells_ll_p0, 1, 1, (gridtools::array<uint_t,3>{9, 24, 25}));
    EVAL_C(cell2cells_ll_p0, 1, 2, (gridtools::array<uint_t,3>{10, 25, 26}));
    EVAL_C(cell2cells_ll_p1, 1, 3, (gridtools::array<uint_t,3>{19, 20, 35}));
    EVAL_C(cell2cells_ll_p1, 1, 4, (gridtools::array<uint_t,3>{20, 21, 36}));
    _EVAL_C(cells,cells, 2, 3, (gridtools::array<uint_t,3>{33, 34, 49}));
    _EVAL_C(cells,cells, 2, 4, (gridtools::array<uint_t,3>{26, 41, 42}));
    _EVAL_C(cells,cells, 3, 3, (gridtools::array<uint_t,3>{49, 50, 65}));
    _EVAL_C(cells,cells, 3, 4, (gridtools::array<uint_t,3>{42, 57, 58}));

    EVAL_C(edge2edges_ll_p0, 2, 3, (gridtools::array<uint_t,4>{66,67,59,82}));
    EVAL_C(edge2edges_ll_p1, 2, 3, (gridtools::array<uint_t,4>{43,28,51,67}));
    EVAL_C(edge2edges_ll_p2, 2, 3, (gridtools::array<uint_t,4>{51,59,52,83}));
    _EVAL_C(edges,edges, 2, 2, (gridtools::array<uint_t,4>{48,56,49,80}));
    _EVAL_C(edges,edges, 2, 3, (gridtools::array<uint_t,4>{64,57,65,80}));
    _EVAL_C(edges,edges, 2, 4, (gridtools::array<uint_t,4>{41,26,49,65}));

    EVAL_C(cell2edges_ll_p0, 2, 3, (gridtools::array<uint_t,3>{51,59,67}));
    EVAL_C(cell2edges_ll_p1, 2, 3, (gridtools::array<uint_t,3>{67,52,83}));
    _EVAL_C(cells,edges, 2, 3, (gridtools::array<uint_t,3>{65,50,81}));
    _EVAL_C(cells,edges, 2, 4, (gridtools::array<uint_t,3>{58,50,66}));

    EVAL_C(edge2cells_ll_p0, 2, 3, (gridtools::array<uint_t,2>{42,35}));
    EVAL_C(edge2cells_ll_p1, 2, 3, (gridtools::array<uint_t,2>{27,35}));
    EVAL_C(edge2cells_ll_p2, 2, 3, (gridtools::array<uint_t,2>{35,43}));
    _EVAL_C(edges,cells, 2, 3, (gridtools::array<uint_t,2>{33,40}));
    _EVAL_C(edges,cells, 2, 4, (gridtools::array<uint_t,2>{25,33}));
    _EVAL_C(edges,cells, 2, 5, (gridtools::array<uint_t,2>{33,41}));




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


    cell_storage_type cells_out(product(trapezoid_2D::u_cell_size(gridtools::array<uint_t, 2>{NC, MC})));
    edge_storage_type edges_out(product(trapezoid_2D::u_edge_size(gridtools::array<uint_t, 2>{NE, ME})));



    typedef arg<0, trapezoid_2D::cells> out_cells;
    typedef arg<1, trapezoid_2D::cells> in_cells;
    typedef arg<2, trapezoid_2D::edges> out_edges;
    typedef arg<3, trapezoid_2D::edges> in_edges;

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

        accessor_type<boost::mpl::vector<in_cells, out_cells, out_edges, in_edges>,
                      trapezoid_2D, trapezoid_2D::cells> acc
            (ptrs, grid, 0,0);


        struct _coords {
            int lb0, ub0;
            int lb1, ub1;

            _coords(int lb0, int ub0, int lb1, int ub1)
                : lb0(lb0)
                , ub0(ub0)
                , lb1(lb1)
                , ub1(ub1)
            {}
        } coords(1, NC-1-1, 2, MC-2-1); // closed intervals

        gridtools::colored_backend::run(acc, x, coords);

    }

    {
        auto x = make_esf<stencil_on_edges, trapezoid_2D, trapezoid_2D::edges>
            (out_cells(), out_cells(), out_edges(), in_edges());

        accessor_type<boost::mpl::vector<in_cells, out_cells, out_edges, in_edges>,
                      trapezoid_2D, trapezoid_2D::edges> acc
            (boost::fusion::vector<cell_storage_type*, cell_storage_type*, edge_storage_type*, edge_storage_type*>
             (&cells_out, &cells, &edges_out, &edges), grid, 0,0);


        struct _coords {
            int lb0, ub0;
            int lb1, ub1;

            _coords(int lb0, int ub0, int lb1, int ub1)
                : lb0(lb0)
                , ub0(ub0)
                , lb1(lb1)
                , ub1(ub1)
            {}
        } coords(1, NE-1-1, 3, ME-3-1);

        gridtools::colored_backend::run(acc, x, coords);

    }

    {
        auto x = make_esf<stencil_on_edges_cells, trapezoid_2D, trapezoid_2D::cells>
            (out_cells(), in_cells(), out_edges(), in_edges());

        accessor_type<boost::mpl::vector<in_cells, out_cells, out_edges, in_edges>,
                      trapezoid_2D, trapezoid_2D::cells> acc
            (boost::fusion::vector<cell_storage_type*, cell_storage_type*, edge_storage_type*, edge_storage_type*>
             (&cells_out, &cells, &edges_out, &edges), grid, 0,0);


        struct _coords {
            int lb0, ub0;
            int lb1, ub1;

            _coords(int lb0, int ub0, int lb1, int ub1)
                : lb0(lb0)
                , ub0(ub0)
                , lb1(lb1)
                , ub1(ub1)
            {}
        } coords(1, NE-1-1, 3, ME-3-1);

        gridtools::colored_backend::run(acc, x, coords);

    }

    {
        auto x = make_esf<stencil_on_cells_edges, trapezoid_2D, trapezoid_2D::cells>
            (out_cells(), in_cells(), out_edges(), in_edges());

        accessor_type<boost::mpl::vector<in_cells, out_cells, out_edges, in_edges>,
                      trapezoid_2D, trapezoid_2D::cells> acc
            (boost::fusion::vector<cell_storage_type*, cell_storage_type*, edge_storage_type*, edge_storage_type*>
             (&cells_out, &cells, &edges_out, &edges), grid, 0,0);


        struct _coords {
            int lb0, ub0;
            int lb1, ub1;

            _coords(int lb0, int ub0, int lb1, int ub1)
                : lb0(lb0)
                , ub0(ub0)
                , lb1(lb1)
                , ub1(ub1)
            {}
        } coords(1, NC-1-1, 3, MC-3-1);

        gridtools::colored_backend::run(acc, x, coords);
    }

    std::cout << std::endl;
    return 0;
}
