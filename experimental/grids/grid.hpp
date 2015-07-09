#pragma once

#include <common/array.h>
#include <cassert>
#include <boost/mpl/vector.hpp>
#include "virtual_storage.hpp"
#include "location_type.hpp"
#include "backend.hpp"

namespace gridtools {

    /**
    */
    template <typename Backend>
    class trapezoid_2D_colored {
    public :
        using cells = location_type<0>;
        using edges = location_type<1>;

        template <typename T>
        struct pointer_to;

        template <int I>
        struct pointer_to<location_type<I>> {
            using type = double*;
        };

        using cell_storage_t = typename Backend::template storage_type<cells>;
        using edge_storage_t = typename Backend::template storage_type<edges>;

    private:
        using v_cell_storage_t = virtual_storage<typename cell_storage_t::layout>;
        using v_edge_storage_t = virtual_storage<typename edge_storage_t::layout>;

        const uint_t M,N; // Sizes as cells in a multi-dimensional Cell array

        static constexpr int Dims = 2;

        v_cell_storage_t m_v_cell_storage;
        v_edge_storage_t m_v_edge_storage;
        using virtual_storage_types = typename boost::fusion::vector<v_cell_storage_t*, v_edge_storage_t*>;
        using storage_types = typename boost::mpl::vector<cell_storage_t*, edge_storage_t*>;
        virtual_storage_types m_virtual_storages;
    public:
        template <typename T>
        struct virtual_storage_type;

        template <int I>
        struct virtual_storage_type<location_type<I> > {
            using type = typename boost::fusion::result_of::at_c<virtual_storage_types, I>::type;
        };

        template <typename T>
        struct storage_type;

        template <int I>
        struct storage_type<location_type<I> > {
            using type = typename boost::mpl::at_c<storage_types, I>::type;
        };


        static constexpr uint_t u_cell_size_j(int _M) {return _M+4;}
        static constexpr uint_t u_cell_size_i(int _N) {return _N+2;}
        static constexpr uint_t u_edge_size_j(int _M) {return 3*(_M/2)+6;}
        static constexpr uint_t u_edge_size_i(int _N) {return _N+2;}

        /** i[1] must be even */
        static /*constexpr*/ array<uint_t, Dims+1> u_cell_size(array<uint_t, Dims> const& i) {
            return array<uint_t, Dims+1>{i[0], 2, i[1]/2};
        }

        /** i[1] must be even */
        static /*constexpr*/ array<uint_t, Dims+1> u_edge_size(array<uint_t, Dims> const& i) {
            return array<uint_t, Dims+1>{i[0]+1, 3, (i[1])/3};
        }

    private:
        int cell_size_j() const {return static_cast<int>(u_cell_size_j(M));}
        int cell_size_i() const {return static_cast<int>(u_cell_size_i(M));}
        int edge_size_j() const {return static_cast<int>(u_edge_size_j(M));}
        int edge_size_i() const {return static_cast<int>(u_edge_size_i(M));}

        uint_t u_cell_size_j() const {return (u_cell_size_j(M));}
        uint_t u_cell_size_i() const {return (u_cell_size_i(M));}
        uint_t u_edge_size_j() const {return (u_edge_size_j(M));}
        uint_t u_edge_size_i() const {return (u_edge_size_i(M));}

        trapezoid_2D_colored() = delete;
    public :

        trapezoid_2D_colored(uint_t N, uint_t M)
            : M{M}
            , N{N}
            , m_v_cell_storage(u_cell_size({u_cell_size_i(N),u_cell_size_j(M)}))
            , m_v_edge_storage(u_edge_size({u_edge_size_i(N),u_edge_size_j(M)}))
        {
            boost::fusion::at_c<cells::value>(m_virtual_storages) = &m_v_cell_storage;
            boost::fusion::at_c<edges::value>(m_virtual_storages) = &m_v_edge_storage;
            assert((M&1) == 0);
            assert((N&1) == 0);
        }

        virtual_storage_types const& virtual_storages() const {return m_virtual_storages;}

#define DO_THE_MATH(stor, i,j,k)                \
        m_v_ ## stor ## _storage._index(i,j,k)


        array<int_t, 3> ll_indices(array<int_t, 2> const& i, cells) const {
            // std::cout << " *cells* " << std::endl;
            return array<int_t, 3>{i[0], i[1]%2, i[1]/2};
        }

        array<int_t, 3> ll_indices(array<int_t, 2> const& i, edges) const {
            // std::cout << " *edges* " << std::endl;
            return array<int_t, 3>{i[0], i[1]%3, i[1]/3};
        }

        int_t ll_offset(array<uint_t, 3> const& i, cells) const {
#ifdef _GRID_H_DEBUG
            std::cout << " **cells offsets** "
                      << m_v_cell_storage._index(i[0], i[1], i[2]) << " from ("
                      << i[0] << ", "
                      << i[1] << ", "
                      << i[2] << ")"
                      << std::endl;
#endif
            return m_v_cell_storage._index(i[0], i[1], i[2]);
        }

        int_t ll_offset(array<uint_t, 3> const& i, edges) const {
#ifdef _GRID_H_DEBUG
            std::cout << " **edges offsets** "
                      << m_v_cell_storage._index(i[0], i[1], i[2]) << " from ("
                      << i[0] << ", "
                      << i[1] << ", "
                      << i[2] << ")"
                      << std::endl;
#endif
            return m_v_edge_storage._index(i[0], i[1], i[2]);
        }

        array<int_t, 3>
        cell2cells_ll_p1(array<int_t, 2> const& i) const
        {
            return array<int_t, 3>{
                DO_THE_MATH(cell, i[0], 0, i[1]),
                DO_THE_MATH(cell, i[0], 0, i[1]+1),
                DO_THE_MATH(cell, i[0]+1, 0, i[1])};
        }

        array<int_t, 3>
        cell2cells_ll_p0(array<int_t, 2> const& i) const
        {
            return array<int_t, 3>{
                DO_THE_MATH(cell, i[0], 1, i[1]-1),
                DO_THE_MATH(cell, i[0], 1, i[1]),
                DO_THE_MATH(cell, i[0]-1, 1, i[1])};
        }

        array<int_t, 4>
        edge2edges_ll_p0(array<int_t, 2> const& i) const
        {
            return array<int_t, 4>{
                DO_THE_MATH(edge, i[0], 1, i[1]),
                DO_THE_MATH(edge, i[0]+1, 1, i[1]-1),
                DO_THE_MATH(edge, i[0], 2, i[1]),
                DO_THE_MATH(edge, i[0], 2, i[1]-1)};
        }

        array<int_t, 4>
        edge2edges_ll_p1(array<int_t, 2> const& i) const
        {
            return array<int_t, 4>{
                DO_THE_MATH(edge, i[0], 0, i[1]),
                DO_THE_MATH(edge, i[0]-1, 0, i[1]+1),
                DO_THE_MATH(edge, i[0], 2, i[1]),
                DO_THE_MATH(edge, i[0]-1, 2, i[1])};
        }

        array<int_t, 4>
        edge2edges_ll_p2(array<int_t, 2> const& i) const
        {
            return array<int_t, 4>{
                DO_THE_MATH(edge, i[0], 0, i[1]),
                DO_THE_MATH(edge, i[0], 0, i[1]+1),
                DO_THE_MATH(edge, i[0], 1, i[1]),
                DO_THE_MATH(edge, i[0]+1, 1, i[1])};
        }

        array<int_t, 3>
        cell2edges_ll_p1(array<int_t, 2> const& i) const
        {
            return array<int_t, 3>{
                DO_THE_MATH(edge, i[0], 2, i[1]),
                DO_THE_MATH(edge, i[0], 0, i[1]+1),
                DO_THE_MATH(edge, i[0]+1, 1, i[1])};
        }

        array<int_t, 3>
        cell2edges_ll_p0(array<int_t, 2> const& i) const
        {
            return array<int_t, 3>{
                DO_THE_MATH(edge, i[0], 0, i[1]),
                DO_THE_MATH(edge, i[0], 1, i[1]),
                DO_THE_MATH(edge, i[0], 2, i[1])};
        }

        array<int_t, 2>
        edge2cells_ll_p0(array<int_t, 2> const& i) const
        {
            return array<int_t, 2>{
                DO_THE_MATH(cell, i[0], 1, i[1]-1),
                DO_THE_MATH(cell, i[0], 0, i[1])};
        }

        array<int_t, 2>
        edge2cells_ll_p1(array<int_t, 2> const& i) const
        {
            return array<int_t, 2>{
                DO_THE_MATH(cell, i[0]-1, 1, i[1]),
                DO_THE_MATH(cell, i[0], 0, i[1])};
        }

        array<int_t, 2>
        edge2cells_ll_p2(array<int_t, 2> const& i) const
        {
            return array<int_t, 2>{
                DO_THE_MATH(cell, i[0], 0, i[1]),
                DO_THE_MATH(cell, i[0], 1, i[1])};
        }

        array<int_t, 3>
        neighbors(array<int_t, 2> const& i, cells, cells) const
        {
            // std::cout << "grid.neighbors cells->cells "
            //           << i[0] << ", "
            //           << i[1]
            //           << std::endl;
            if (i[1]&1) {
                return cell2cells_ll_p1({i[0], i[1]/2});
            } else {
                return cell2cells_ll_p0({i[0], i[1]/2});
            }
        }

        array<int_t, 4>
        neighbors(array<int_t, 2> const& i, edges, edges) const
        {
            // std::cout << "grid.neighbors edges->edges "
            //           << i[0] << ", "
            //           << i[1]
            //           << std::endl;
            switch (i[1]%3) {
            case 0:
                return edge2edges_ll_p0({i[0], i[1]/3});
            case 1:
                return edge2edges_ll_p1({i[0], i[1]/3});
            case 2:
                return edge2edges_ll_p2({i[0], i[1]/3});
            }
        }

        array<int_t, 3>
        neighbors(array<int_t, 2> const& i, cells, edges) const
        {
            // std::cout << "grid.neighbors cells->edges "
            //           << i[0] << ", "
            //           << i[1]
            //           << std::endl;
            if (i[1]&1) {
                return cell2edges_ll_p1({i[0], i[1]/2});
            } else {
                return cell2edges_ll_p0({i[0], i[1]/2});
            }
        }

        array<int_t, 2>
        neighbors(array<int_t, 2> const& i, edges, cells) const
        {
            // std::cout << "grid.neighbors edges->cells "
            //           << i[0] << ", "
            //           << i[1]
            //           << std::endl;
            switch (i[1]%3) {
            case 0:
                return edge2cells_ll_p0({i[0], i[1]/3});
            case 1:
                return edge2cells_ll_p1({i[0], i[1]/3});
            case 2:
                return edge2cells_ll_p2({i[0], i[1]/3});
            }
        }



        /////////////////////////////////////////////////////////////////////
        array<int_t, 3>
        neighbors_ll(array<int_t, 3> const& i, cells, cells) const
        {
            // std::cout << "grid.neighbors cells->cells "
            //           << i[0] << ", "
            //           << i[1]
            //           << std::endl;
            if (i[1]&1) {
                return cell2cells_ll_p1({i[0], i[2]});
            } else {
                return cell2cells_ll_p0({i[0], i[2]});
            }
        }

        array<int_t, 4>
        neighbors_ll(array<int_t, 3> const& i, edges, edges) const
        {
            // std::cout << "grid.neighbors edges->edges "
            //           << i[0] << ", "
            //           << i[1]
            //           << std::endl;
            switch (i[1]%3) {
            case 0:
                return edge2edges_ll_p0({i[0], i[2]});
            case 1:
                return edge2edges_ll_p1({i[0], i[2]});
            case 2:
                return edge2edges_ll_p2({i[0], i[2]});
            }
        }

        array<int_t, 3>
        neighbors_ll(array<int_t, 3> const& i, cells, edges) const
        {
            // std::cout << "grid.neighbors cells->edges "
            //           << i[0] << ", "
            //           << i[1]
            //           << std::endl;
            if (i[1]&1) {
                return cell2edges_ll_p1({i[0], i[2]});
            } else {
                return cell2edges_ll_p0({i[0], i[2]});
            }
        }

        array<int_t, 2>
        neighbors_ll(array<int_t, 3> const& i, edges, cells) const
        {
            // std::cout << "grid.neighbors edges->cells "
            //           << i[0] << ", "
            //           << i[1]
            //           << std::endl;
            switch (i[1]%3) {
            case 0:
                return edge2cells_ll_p0({i[0], i[2]});
            case 1:
                return edge2cells_ll_p1({i[0], i[2]});
            case 2:
                return edge2cells_ll_p2({i[0], i[2]});
            }
        }

        ///////////////////////////////////


        array<array<uint_t, 3>, 3>
        cell2cells_ll_p1_indices(array<uint_t, 2> const& i) const
        {
            return array<array<uint_t, 3>, 3>{
                { i[0], 0, i[1]},
                { i[0], 0, i[1]+1},
                { i[0]+1, 0, i[1]}};
        }

        array<array<uint_t, 3>, 3>
        cell2cells_ll_p0_indices(array<uint_t, 2> const& i) const
        {
            assert(i[1] > 0);
            return array<array<uint_t, 3>, 3>{
                { i[0], 1, i[1]-1},
                { i[0], 1, i[1]},
                { i[0]-1, 1, i[1]}};
        }

        array<array<uint_t, 3>, 4>
        edge2edges_ll_p0_indices(array<uint_t, 2> const& i) const
        {
            assert(i[1] > 0);
            return array<array<uint_t, 3>, 4>{
                { i[0], 1, i[1]},
                { i[0]+1, 1, i[1]-1},
                { i[0], 2, i[1]},
                { i[0], 2, i[1]-1}};
        }

        array<array<uint_t, 3>, 4>
        edge2edges_ll_p1_indices(array<uint_t, 2> const& i) const
        {
            assert(i[0] > 0);
            return array<array<uint_t, 3>, 4>{
                { i[0], 0, i[1]},
                { i[0]-1, 0, i[1]+1},
                { i[0], 2, i[1]},
                { i[0]-1, 2, i[1]}};
        }

        array<array<uint_t, 3>, 4>
        edge2edges_ll_p2_indices(array<uint_t, 2> const& i) const
        {
            return array<array<uint_t, 3>, 4>{
                { i[0], 0, i[1]},
                { i[0], 0, i[1]+1},
                { i[0], 1, i[1]},
                { i[0]+1, 1, i[1]}};
        }

        array<array<uint_t, 3>, 3>
        cell2edges_ll_p1_indices(array<uint_t, 2> const& i) const
        {
#ifdef _GRID_H_DEBUG
            std::cout << "cell2edges_ll_p1_indices " << i[0] << ", " << i[1] << std::endl;
#endif
            return array<array<uint_t, 3>, 3>{
                { i[0], 2, i[1]},
                { i[0], 0, i[1]+1},
                { i[0]+1, 1, i[1]}};
        }

        array<array<uint_t, 3>, 3>
        cell2edges_ll_p0_indices(array<uint_t, 2> const& i) const
        {
#ifdef _GRID_H_DEBUG
            std::cout << "cell2edges_ll_p0_indices " << i[0] << ", " << i[1] << std::endl;
#endif
            return array<array<uint_t, 3>, 3>{
                { i[0], 0, i[1]},
                { i[0], 1, i[1]},
                { i[0], 2, i[1]}};
        }

        array<array<uint_t, 3>, 2>
        edge2cells_ll_p0_indices(array<uint_t, 2> const& i) const
        {
#ifdef _GRID_H_DEBUG
            std::cout << "edge2cells_ll_p0_indices " << i[0] << " " << i[1] << std::endl;
#endif
            assert(i[1] > 0);
            return array<array<uint_t, 3>, 2>{
                { i[0], 1, i[1]-1},
                { i[0], 0, i[1]}};
        }

        array<array<uint_t, 3>, 2>
        edge2cells_ll_p1_indices(array<uint_t, 2> const& i) const
        {
#ifdef _GRID_H_DEBUG
            std::cout << "edge2cells_ll_p1_indices " << i[0] << " " << i[1] << std::endl;
#endif
            assert(i[0] > 0);
            return array<array<uint_t, 3>, 2>{
                { i[0]-1, 1, i[1]},
                { i[0], 0, i[1]}};
        }

        array<array<uint_t, 3>, 2>
        edge2cells_ll_p2_indices(array<uint_t, 2> const& i) const
        {
#ifdef _GRID_H_DEBUG
            std::cout << "edge2cells_ll_p2_indices " << i[0] << " " << i[1] << std::endl;
#endif
            return array<array<uint_t, 3>, 2>{
                { i[0], 0, i[1]},
                { i[0], 1, i[1]}};
        }

        array<array<uint_t, 3>, 3>
        neighbors_indices(array<uint_t, 2> const& i, cells, cells) const
        {
            if (i[1]&1) {
                return cell2cells_ll_p1_indices({i[0], i[1]/2});
            } else {
                return cell2cells_ll_p0_indices({i[0], i[1]/2});
            }
        }

        array<array<uint_t, 3>, 4>
        neighbors_indices(array<uint_t, 2> const& i, edges, edges) const
        {
            switch (i[1]%3) {
            case 0:
                return edge2edges_ll_p0_indices({i[0], i[1]/3});
            case 1:
                return edge2edges_ll_p1_indices({i[0], i[1]/3});
            case 2:
                return edge2edges_ll_p2_indices({i[0], i[1]/3});
            }
        }

        array<array<uint_t, 3>, 3>
        neighbors_indices(array<uint_t, 2> const& i, cells, edges) const
        {
            if (i[1]&1) {
                return cell2edges_ll_p1_indices({i[0], i[1]/2});
            } else {
                return cell2edges_ll_p0_indices({i[0], i[1]/2});
            }
        }

        array<array<uint_t, 3>, 2>
        neighbors_indices(array<uint_t, 2> const& i, edges, cells) const
        {
            switch (i[1]%3) {
            case 0:
                return edge2cells_ll_p0_indices({i[0], i[1]/3});
            case 1:
                return edge2cells_ll_p1_indices({i[0], i[1]/3});
            case 2:
                return edge2cells_ll_p2_indices({i[0], i[1]/3});
            }
        }


        /**************************************************************************/
        array<array<uint_t, 3>, 3>
        neighbors_indices_3(array<uint_t, 3> const& i, cells, cells) const
        {
#ifdef _GRID_H_DEBUG
            std::cout << "neighbors_indices_3 cells cells "
                      << i[0] << ", " << i[1] << ", " << i[2]
                      << std::endl;
#endif
            if (i[1]&1) {
                return cell2cells_ll_p1_indices({i[0], i[2]});
            } else {
                return cell2cells_ll_p0_indices({i[0], i[2]});
            }
        }

        array<array<uint_t, 3>, 4>
        neighbors_indices_3(array<uint_t, 3> const& i, edges, edges) const
        {
#ifdef _GRID_H_DEBUG
            std::cout << "neighbors_indices_3 edges edges "
                      << i[0] << ", " << i[1] << ", " << i[2]
                      << std::endl;
#endif
            switch (i[1]%3) {
            case 0:
                return edge2edges_ll_p0_indices({i[0], i[2]});
            case 1:
                return edge2edges_ll_p1_indices({i[0], i[2]});
            case 2:
                return edge2edges_ll_p2_indices({i[0], i[2]});
            }
        }

        array<array<uint_t, 3>, 3>
        neighbors_indices_3(array<uint_t, 3> const& i, cells, edges) const
        {
#ifdef _GRID_H_DEBUG
            std::cout << "neighbors_indices_3 cells edges "
                      << i[0] << ", " << i[1] << ", " << i[2]
                      << std::endl;
#endif
            if (i[1]&1) {
                return cell2edges_ll_p1_indices({i[0], i[2]});
            } else {
                return cell2edges_ll_p0_indices({i[0], i[2]});
            }
        }

        array<array<uint_t, 3>, 2>
        neighbors_indices_3(array<uint_t, 3> const& i, edges, cells) const
        {
#ifdef _GRID_H_DEBUG
            std::cout << "neighbors_indices_3 edges cells "
                      << i[0] << ", " << i[1] << ", " << i[2]
                      << std::endl;
#endif
            switch (i[1]%3) {
            case 0:
                return edge2cells_ll_p0_indices({i[0], i[2]});
            case 1:
                return edge2cells_ll_p1_indices({i[0], i[2]});
            case 2:
                return edge2cells_ll_p2_indices({i[0], i[2]});
            }
        }


    };

}
