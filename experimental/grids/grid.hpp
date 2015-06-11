#pragma once

#include <common/array.h>
#include <cassert>
#include <boost/mpl/vector.hpp>
#include "virtual_storage.hpp"

namespace gridtools {

    template <int I>
    struct location_type {
        static const int value = I;
    };

    template <int I>
    std::ostream& operator<<(std::ostream& s, location_type<I>) {
        return s << "location_type<" << I << ">";
    }

    /**
    */
    template <typename CellStorageT, typename EdgeStorageT>
    class trapezoid_2D_no_tile {
    public :
        using cells = location_type<0>;
        using edges = location_type<1>;

        template <typename T>
        struct pointer_to;

        template <int I>
        struct pointer_to<location_type<I>> {
            using type = double*;
        };

    private:
        using cell_storage_t = CellStorageT;
        using edge_storage_t = EdgeStorageT;

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

        trapezoid_2D_no_tile() = delete;
    public :

        trapezoid_2D_no_tile(uint_t N, uint_t M)
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

        
        array<uint_t, 3>
        cell2cells_ll_p1(array<uint_t, 2> const& i) const
        {
            return array<uint_t, 3>{
                DO_THE_MATH(cell, i[0], 0, i[1]),
                DO_THE_MATH(cell, i[0], 0, i[1]+1),
                DO_THE_MATH(cell, i[0]+1, 0, i[1])};
        }

        array<uint_t, 3>
        cell2cells_ll_p0(array<uint_t, 2> const& i) const
        {
            return array<uint_t, 3>{
                DO_THE_MATH(cell, i[0], 1, i[1]-1),
                DO_THE_MATH(cell, i[0], 1, i[1]),
                DO_THE_MATH(cell, i[0]-1, 1, i[1])};
        }

        array<uint_t, 4>
        edge2edges_ll_p0(array<uint_t, 2> const& i) const
        {
            return array<uint_t, 4>{
                DO_THE_MATH(edge, i[0], 1, i[1]),
                DO_THE_MATH(edge, i[0]+1, 1, i[1]-1),
                DO_THE_MATH(edge, i[0], 2, i[1]),
                DO_THE_MATH(edge, i[0], 2, i[1]-1)};
        }

        array<uint_t, 4>
        edge2edges_ll_p1(array<uint_t, 2> const& i) const
        {
            return array<uint_t, 4>{
                DO_THE_MATH(edge, i[0], 0, i[1]),
                DO_THE_MATH(edge, i[0]-1, 0, i[1]+1),
                DO_THE_MATH(edge, i[0], 2, i[1]),
                DO_THE_MATH(edge, i[0]-1, 2, i[1])};
        }

        array<uint_t, 4>
        edge2edges_ll_p2(array<uint_t, 2> const& i) const
        {
            return array<uint_t, 4>{
                DO_THE_MATH(edge, i[0], 0, i[1]),
                DO_THE_MATH(edge, i[0], 0, i[1]+1),
                DO_THE_MATH(edge, i[0], 1, i[1]),
                DO_THE_MATH(edge, i[0]+1, 1, i[1])};
        }

        array<uint_t, 3>
        cell2edges_ll_p1(array<uint_t, 2> const& i) const
        {
            return array<uint_t, 3>{
                DO_THE_MATH(edge, i[0], 2, i[1]),
                DO_THE_MATH(edge, i[0], 0, i[1]+1),
                DO_THE_MATH(edge, i[0]+1, 1, i[1])};
        }

        array<uint_t, 3>
        cell2edges_ll_p0(array<uint_t, 2> const& i) const
        {
            return array<uint_t, 3>{
                DO_THE_MATH(edge, i[0], 0, i[1]),
                DO_THE_MATH(edge, i[0], 1, i[1]),
                DO_THE_MATH(edge, i[0], 2, i[1])};
        }

        array<uint_t, 2>
        edge2cells_ll_p0(array<uint_t, 2> const& i) const
        {
            return array<uint_t, 2>{
                DO_THE_MATH(cell, i[0], 1, i[1]-1),
                DO_THE_MATH(cell, i[0], 0, i[1])};
        }

        array<uint_t, 2>
        edge2cells_ll_p1(array<uint_t, 2> const& i) const
        {
            return array<uint_t, 2>{
                DO_THE_MATH(cell, i[0]-1, 1, i[1]),
                DO_THE_MATH(cell, i[0], 0, i[1])};
        }

        array<uint_t, 2>
        edge2cells_ll_p2(array<uint_t, 2> const& i) const
        {
            return array<uint_t, 2>{
                DO_THE_MATH(cell, i[0], 0, i[1]),
                DO_THE_MATH(cell, i[0], 1, i[1])};
        }

        array<uint_t, 3>
        cell2cells(array<uint_t, 2> const& i) const
        {
            if (i[1]&1) {
                return cell2cells_ll_p1({i[0], i[1]/2});
            } else {
                return cell2cells_ll_p0({i[0], i[1]/2});
            }
        }        

        array<uint_t, 4>
        edge2edges(array<uint_t, 2> const& i) const
        {
            switch (i[1]%3) {
            case 0:
                return edge2edges_ll_p0({i[0], i[1]/3});
            case 1:
                return edge2edges_ll_p1({i[0], i[1]/3});
            case 2:
                return edge2edges_ll_p2({i[0], i[1]/3});
            }
        }

        array<uint_t, 3>
        cell2edges(array<uint_t, 2> const& i) const
        {
            if (i[1]&1) {
                return cell2edges_ll_p1({i[0], i[1]/2});
            } else {
                return cell2edges_ll_p0({i[0], i[1]/2});
            }
        }

        array<uint_t, 2>
        edge2cells(array<uint_t, 2> const& i) const
        {
            switch (i[1]%3) {
            case 0:
                return edge2cells_ll_p0({i[0], i[1]/3});
            case 1:
                return edge2cells_ll_p1({i[0], i[1]/3});
            case 2:
                return edge2cells_ll_p2({i[0], i[1]/3});
            }
        }

    };

}

