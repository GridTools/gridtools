/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
using uint_t = unsigned int;
using int_t = int;
#include "array_addons.hpp"
#include "grid.hpp"
#include "iterate_domain.hpp"
#include "make_stencil.hpp"
#include "placeholders.hpp"
#include <boost/fusion/adapted/mpl/detail/size_impl.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/vector.hpp>
#include <common/layout_map.hpp>
#include <iostream>
#include <storage/base_storage.hpp>
#include <type_traits>

using gridtools::layout_map;
using gridtools::wrap_pointer;

using trapezoid_2D = gridtools::trapezoid_2D_colored< gridtools::_backend >;

using cell_storage_type = typename trapezoid_2D::storage_t< trapezoid_2D::cells, double >;
using edge_storage_type = typename trapezoid_2D::storage_t< trapezoid_2D::edges, double >;
using vertex_storage_type = typename trapezoid_2D::storage_t< trapezoid_2D::vertexes, double >;

struct stencil_on_cells {
    typedef accessor< 0, trapezoid_2D::cells > out;
    typedef ro_accessor< 1, trapezoid_2D::cells, radius< 1 > > in;
    typedef ro_accessor< 2, trapezoid_2D::edges > out_edges_NOT_USED;
    typedef ro_accessor< 3, trapezoid_2D::edges, radius< 1 > > in_edges;

    template < typename GridAccessors >
    void operator()(GridAccessors /*const*/ &eval /*, region*/) const {
        // std::cout << "i = " << eval.i()
        //           << " j = " << eval.j()
        //           << std::endl;
        auto ff = [](const double _in, const double _res) -> double {
#ifdef _MAIN_CPP_DEBUG_
            std::cout << "#";
#endif
            return _in + _res;
        };

        /**
           This interface checks that the location types are compatible with the accessors
         */
        eval(out()) = eval(reduce_on_cells(ff, 0.0, in())) + eval(reduce_on_edges(ff, 0.0, in_edges()));
    }
};

struct stencil_on_vertexes {
    typedef accessor< 0, trapezoid_2D::vertexes > out;
    typedef ro_accessor< 1, trapezoid_2D::vertexes, radius< 1 > > in;

    template < typename GridAccessors >
    void operator()(GridAccessors /*const*/ &eval /*, region*/) const {
        auto ff = [](const double _in, const double _res) -> double { return _in + _res; };

        /**
           This interface checks that the location types are compatible with the accessors
         */
        eval(out()) = eval(reduce_on_vertexes(ff, 0.0, in()));
    }
};

#define _MAIN_CPP_DEBUG_

struct nested_stencil {
    typedef accessor< 0, trapezoid_2D::cells > out;
    typedef ro_accessor< 1, trapezoid_2D::cells, radius< 2 > > in;
    typedef ro_accessor< 2, trapezoid_2D::edges, radius< 1 > > edges0;
    typedef accessor< 3, trapezoid_2D::edges > edges1;

    template < typename GridAccessors >
    void operator()(GridAccessors /*const*/ &eval /*, region*/) const {
        // std::cout << "i = " << eval.i()
        //           << " j = " << eval.j()
        //           << std::endl;
        auto ff = [](const double _in, const double _res) -> double {
#ifdef _MAIN_CPP_DEBUG_
            std::cout << "#";
#endif
            return _in + _res + 1;
        };
        auto gg = [](const double _in, const double _res) -> double {
#ifdef _MAIN_CPP_DEBUG_
            std::cout << "m";
#endif
            return _in + _res + 2;
        };
        auto reduction = [](const double _in, const double _res) -> double {
#ifdef _MAIN_CPP_DEBUG_
            std::cout << "r";
#endif
            return _in + _res + 3;
        };

        auto x = eval(reduce_on_edges(
            reduction, 0.0, map(gg, edges0(), reduce_on_cells(ff, 0.0, map(identity< double >(), in())))));
        auto y = eval(reduce_on_edges(reduction, 0.0, map(gg, edges0(), reduce_on_cells(ff, 0.0, in()))));
        assert(x == y);
        std::cout << x << " == " << y << std::endl;
        // eval(out()) = eval(reduce_on_edges(reduction, 0.0, edges0::reduce_on_cells(gg, in()), edges1()));
    }
};

struct stencil_on_edges {
    typedef accessor< 0, trapezoid_2D::cells > out_NOT_USED;
    typedef ro_accessor< 1, trapezoid_2D::cells, radius< 1 > > in;
    typedef accessor< 2, trapezoid_2D::edges > out_edges;
    typedef ro_accessor< 3, trapezoid_2D::edges, radius< 1 > > in_edges;

    template < typename GridAccessors >
    void operator()(GridAccessors /*const*/ &eval /*, region*/) const {
        // std::cout << "i = " << eval.i()
        //           << " j = " << eval.j()
        //           << std::endl;
        auto ff = [](const double _in, const double _res) -> double {
#ifdef _MAIN_CPP_DEBUG_
            std::cout << "e";
#endif
            return _in + _res;
        };
        eval(out_edges()) = eval(reduce_on_cells(ff, 0.0, in())) + eval(reduce_on_edges(ff, 0.0, in_edges()));
    }
};

template < typename Storage >
void fill_storage_with_indices(Storage &storage) {
    // for(uint_t i =0; i < storage.m_size; ++i)
    //     storage.m_ptr[i] = i;
}

int main() {
    uint_t NC = trapezoid_2D::u_size_i(trapezoid_2D::cells(), 6);
    uint_t MC = trapezoid_2D::u_size_j(trapezoid_2D::cells(), 12);

    uint_t NE = trapezoid_2D::u_size_i(trapezoid_2D::edges(), 6);
    uint_t ME = trapezoid_2D::u_size_j(trapezoid_2D::edges(), 12);

    uint_t NV = trapezoid_2D::u_size_i(trapezoid_2D::vertexes(), 6);
    uint_t MV = trapezoid_2D::u_size_j(trapezoid_2D::vertexes(), 12);

    std::cout << "NC = " << NC << " "
              << "MC = " << MC << std::endl;

    std::cout << "NE = " << NE << " "
              << "ME = " << ME << std::endl;

    uint_t d3 = 11;
    trapezoid_2D grid(6, 12, d3);

    cell_storage_type cells = grid.make_storage< trapezoid_2D::cells >();
    edge_storage_type edges = grid.make_storage< trapezoid_2D::edges >();
    vertex_storage_type vertexes = grid.make_storage< trapezoid_2D::vertexes >();

    fill_storage_with_indices(cells);
    fill_storage_with_indices(edges);
    fill_storage_with_indices(vertexes);

    cell_storage_type cells_out = grid.make_storage< trapezoid_2D::cells >();
    edge_storage_type edges_out = grid.make_storage< trapezoid_2D::edges >();
    vertex_storage_type vertexes_out = grid.make_storage< trapezoid_2D::vertexes >();

    fill_storage_with_indices(cells_out);
    fill_storage_with_indices(edges_out);
    fill_storage_with_indices(vertexes_out);

    typedef arg< 0, trapezoid_2D::cells > out_cells;
    typedef arg< 1, trapezoid_2D::cells > in_cells;
    typedef arg< 2, trapezoid_2D::edges > out_edges;
    typedef arg< 3, trapezoid_2D::edges > in_edges;
    typedef arg< 4, trapezoid_2D::vertexes > out_vertexes;
    typedef arg< 5, trapezoid_2D::vertexes > in_vertexes;

    std::cout << "#####################################################################################################"
                 "########################################################"
              << std::endl;
    std::cout << "#####################################################################################################"
                 "########################################################"
              << std::endl;
    std::cout << "CASE # 1" << std::endl;
    std::cout << "#####################################################################################################"
                 "########################################################"
              << std::endl;
    std::cout << "#####################################################################################################"
                 "########################################################"
              << std::endl;

    {
        auto x = make_stage< stencil_on_cells, trapezoid_2D, trapezoid_2D::cells >(
            out_cells(), in_cells(), out_edges(), in_edges());

        auto ptrs =
            boost::fusion::vector< cell_storage_type *, cell_storage_type *, edge_storage_type *, edge_storage_type * >(
                &cells_out, &cells, &edges_out, &edges);

        iterate_domain< boost::mpl::vector< in_cells, out_cells, out_edges, in_edges >,
            trapezoid_2D,
            trapezoid_2D::cells > acc(ptrs, grid);

        struct _coords {
            int_t lb0, ub0;
            int_t lb1, ub1;
            int_t lb2, ub2;

            _coords(int lb0, int ub0, int lb1, int ub1, int lb2, int ub2)
                : lb0(lb0), ub0(ub0), lb1(lb1), ub1(ub1), lb2(lb2), ub2(ub2) {}
        } coords(1, NC - 1, 2, MC - 2, 0, d3); // closed intervals

        gridtools::colored_backend::run(acc, x, coords);
    }

    std::cout << "#####################################################################################################"
                 "########################################################"
              << std::endl;
    std::cout << "#####################################################################################################"
                 "########################################################"
              << std::endl;
    std::cout << "CASE # 2" << std::endl;
    std::cout << "#####################################################################################################"
                 "########################################################"
              << std::endl;
    std::cout << "#####################################################################################################"
                 "########################################################"
              << std::endl;

    {
        auto x = make_stage< stencil_on_edges, trapezoid_2D, trapezoid_2D::edges >(
            out_cells(), out_cells(), out_edges(), in_edges());

        iterate_domain< boost::mpl::vector< in_cells, out_cells, out_edges, in_edges >,
            trapezoid_2D,
            trapezoid_2D::cells >
        acc(boost::fusion::vector< cell_storage_type *, cell_storage_type *, edge_storage_type *, edge_storage_type * >(
                &cells_out, &cells, &edges_out, &edges),
            grid);

        struct _coords {
            int_t lb0, ub0;
            int_t lb1, ub1;
            int_t lb2, ub2;

            _coords(int lb0, int ub0, int lb1, int ub1, int lb2, int ub2)
                : lb0(lb0), ub0(ub0), lb1(lb1), ub1(ub1), lb2(lb2), ub2(ub2) {}
        } coords(1, NC - 1, 2, MC - 2, 0, d3); // closed intervals

        gridtools::colored_backend::run(acc, x, coords);
    }

    std::cout << "#####################################################################################################"
                 "########################################################"
              << std::endl;
    std::cout << "#####################################################################################################"
                 "########################################################"
              << std::endl;
    std::cout << "CASE # 3" << std::endl;
    std::cout << "#####################################################################################################"
                 "########################################################"
              << std::endl;
    std::cout << "#####################################################################################################"
                 "########################################################"
              << std::endl;

    {
        auto x = make_stage< stencil_on_vertexes, trapezoid_2D, trapezoid_2D::vertexes >(out_vertexes(), in_vertexes());

        auto ptrs = boost::fusion::vector< vertex_storage_type *, vertex_storage_type * >(&vertexes_out, &vertexes);

        iterate_domain< boost::mpl::vector< out_vertexes, in_vertexes >, trapezoid_2D, trapezoid_2D::vertexes > it_(
            ptrs, grid);

        struct _coords {
            int_t lb0, ub0;
            int_t lb1, ub1;
            int_t lb2, ub2;

            _coords(int lb0, int ub0, int lb1, int ub1, int lb2, int ub2)
                : lb0(lb0), ub0(ub0), lb1(lb1), ub1(ub1), lb2(lb2), ub2(ub2) {}
        } coords(1, NC - 1, 2, MC - 2, 0, d3); // closed intervals

        gridtools::colored_backend::run(it_, x, coords);
    }

    std::cout << "#####################################################################################################"
                 "########################################################"
              << std::endl;
    std::cout << "#####################################################################################################"
                 "########################################################"
              << std::endl;
    std::cout << "CASE # 4" << std::endl;
    std::cout << "#####################################################################################################"
                 "########################################################"
              << std::endl;
    std::cout << "#####################################################################################################"
                 "########################################################"
              << std::endl;

    {
        auto x = make_stage< nested_stencil, trapezoid_2D, trapezoid_2D::cells >(
            out_cells(), in_cells(), out_edges(), in_edges());

        iterate_domain< boost::mpl::vector< in_cells, out_cells, out_edges, in_edges >,
            trapezoid_2D,
            trapezoid_2D::cells >
        acc(boost::fusion::vector< cell_storage_type *, cell_storage_type *, edge_storage_type *, edge_storage_type * >(
                &cells_out, &cells, &edges_out, &edges),
            grid);

        struct _coords {
            int_t lb0, ub0;
            int_t lb1, ub1;
            int_t lb2, ub2;

            _coords(int lb0, int ub0, int lb1, int ub1, int lb2, int ub2)
                : lb0(lb0), ub0(ub0), lb1(lb1), ub1(ub1), lb2(lb2), ub2(ub2) {}
        } coords(1, NC - 1, 2, MC - 2, 0, d3); // closed intervals

        gridtools::colored_backend::run(acc, x, coords);
    }

    std::cout << std::endl;
    return 0;
}
