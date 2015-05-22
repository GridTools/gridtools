#pragma once

#include <common/array.h>
#include <cassert>

namespace gridtools {

    /**
    */
    template <typename CellStorageT, typename EdgeStorageT, unsigned int TileI, unsigned int TileJ>
    class trapezoid_2D {
        using cell_storage_t = CellStorageT;
        using edge_storage_t = EdgeStorageT;

        cell_storage_t const& cell_storage;
        edge_storage_t const& edge_storage;
        const unsigned int M,N; // Sizes as cells in a multi-dimensional Cell array

        static constexpr int Dims = 2;

    public:
        static constexpr unsigned int u_cell_size_j(int _M) {return _M+4;}
        static constexpr unsigned int u_cell_size_i(int _N) {return _N+2;}
        static constexpr unsigned int u_edge_size_j(int _M) {return 3*(_M/2)+3;}
        static constexpr unsigned int u_edge_size_i(int _N) {return _N+2;}

    private:
        int cell_size_j() const {return static_cast<int>(u_cell_size_j(M));}
        int cell_size_i() const {return static_cast<int>(u_cell_size_i(M));}
        int edge_size_j() const {return static_cast<int>(u_edge_size_j(M));}
        int edge_size_i() const {return static_cast<int>(u_edge_size_i(M));}

        unsigned int u_cell_size_j() const {return (u_cell_size_j(M));}
        unsigned int u_cell_size_i() const {return (u_cell_size_i(M));}
        unsigned int u_edge_size_j() const {return (u_edge_size_j(M));}
        unsigned int u_edge_size_i() const {return (u_edge_size_i(M));}


        array<unsigned int, cell_storage_t::space_dimensions> cell_api_to_internal(array<unsigned int, Dims> const& indices) const {
            return array<unsigned int, cell_storage_t::space_dimensions>{indices[0]/TileI,
                                                                         indices[1]/TileJ,
                                                                         indices[0]%TileI,
                                                                         indices[1]&1,
                                                                         (indices[1]/2)%(TileJ)
                                                                        };
        }

    public :
        trapezoid_2D(cell_storage_t const& cs, edge_storage_t const& es)
            : cell_storage(cs)
            , edge_storage(es)
            , M{cell_storage.size_j()}
            , N{cell_storage.size_i()}
        {
            assert((M&1) == 0);
            assert((N&1) == 0);
        }

        array<int, 3> cell2cells_offsets(array<unsigned int, Dims> const& indices) const {
            return array<int, 3>{-1, 1, (indices[1]&1)?cell_size_j():-cell_size_j()};
        }

        array<int, 4> edge2edges_offsets(array<unsigned int, Dims> const& indices) const {
            return array<int, 4>{ -1, +1, (indices[1]&1)?-2:2, edge_size_j()+ ((indices[1]&1)?0:1)};
        }

        array<int, 3> cell2edges_offsets(array<unsigned int, Dims> const& indices) const {
            return array<int, 3>{edge_storage.offset({indices[0], (indices[1]/2)+1}), 1, (indices[0]&1)?3:0};
        }

    };

    /**
    */
    template <typename CellStorageT, typename EdgeStorageT>
    class trapezoid_2D_no_tile {
        using cell_storage_t = CellStorageT;
        using edge_storage_t = EdgeStorageT;

        cell_storage_t const* cell_storage;
        edge_storage_t const* edge_storage;
        const unsigned int M,N; // Sizes as cells in a multi-dimensional Cell array

        static constexpr int Dims = 2;

    public:
        static constexpr unsigned int u_cell_size_j(int _M) {return _M+4;}
        static constexpr unsigned int u_cell_size_i(int _N) {return _N+2;}
        static constexpr unsigned int u_edge_size_j(int _M) {return 3*(_M/2)+6;}
        static constexpr unsigned int u_edge_size_i(int _N) {return _N+2;}

        unsigned int cs0;
        unsigned int cs1;
        unsigned int cs2;
        unsigned int ce0;
        unsigned int ce1;
        unsigned int ce2;
        
        /** i[1] must be even */
        /*static constexpr*/ array<unsigned int, Dims+1> u_cell_size(array<unsigned int, Dims> const& i) {
            cs0 = i[0];
            cs1 = 2;
            cs2 = i[1]/2;
            return array<unsigned int, Dims+1>{i[0], 2, i[1]/2};
        }

        /** i[1] must be even */
        /*static constexpr*/ array<unsigned int, Dims+1> u_edge_size(array<unsigned int, Dims> const& i) {
            ce0 = i[0]+1;
            ce1 = 3;
            ce2 = i[1]/3;
            return array<unsigned int, Dims+1>{i[0]+1, 3, (i[1])/3};
        }

    private:
        int cell_size_j() const {return static_cast<int>(u_cell_size_j(M));}
        int cell_size_i() const {return static_cast<int>(u_cell_size_i(M));}
        int edge_size_j() const {return static_cast<int>(u_edge_size_j(M));}
        int edge_size_i() const {return static_cast<int>(u_edge_size_i(M));}

        unsigned int u_cell_size_j() const {return (u_cell_size_j(M));}
        unsigned int u_cell_size_i() const {return (u_cell_size_i(M));}
        unsigned int u_edge_size_j() const {return (u_edge_size_j(M));}
        unsigned int u_edge_size_i() const {return (u_edge_size_i(M));}

        unsigned int cell__index(int i, int j, int k) const {
            return (cs1*cs2)*i+cs2*j+k;
        }
        
        unsigned int edge__index(int i, int j, int k) const {
            return (ce1*ce2)*i+ce2*j+k;
        }
        
    public :
        trapezoid_2D_no_tile()
            : N{}
            , M{}
        {}

        trapezoid_2D_no_tile(cell_storage_t const& cs, edge_storage_t const& es, unsigned int N, unsigned int M)
            : cell_storage(&cs)
            , edge_storage(&es)
            , M{M}
            , N{N}
        {
            u_cell_size({N,M});
            u_edge_size({N,M});
            assert((M&1) == 0);
            assert((N&1) == 0);
        }

#ifdef USE_STORAGE
#define DO_THE_MATH(stor, i,j,k)                \
        stor ## _storage->_index(i,j,k)
#else
#define DO_THE_MATH(stor, i,j,k)                \
        stor ## __index(i,j,k)
#endif
        
        array<unsigned int, 3>
        cell2cells_ll_p1(array<unsigned int, 2> const& i) const
        {
            return array<unsigned int, 3>{
                DO_THE_MATH(cell, i[0], 0, i[1]),
                DO_THE_MATH(cell, i[0], 0, i[1]+1),
                DO_THE_MATH(cell, i[0]+1, 0, i[1])};
        }

        array<unsigned int, 3>
        cell2cells_ll_p0(array<unsigned int, 2> const& i) const
        {
            return array<unsigned int, 3>{
                DO_THE_MATH(cell, i[0], 1, i[1]-1),
                DO_THE_MATH(cell, i[0], 1, i[1]),
                DO_THE_MATH(cell, i[0]-1, 1, i[1])};
        }

        array<unsigned int, 4>
        edge2edges_ll_p0(array<unsigned int, 2> const& i) const
        {
            return array<unsigned int, 4>{
                DO_THE_MATH(edge, i[0], 1, i[1]),
                DO_THE_MATH(edge, i[0]+1, 1, i[1]-1),
                DO_THE_MATH(edge, i[0], 2, i[1]),
                DO_THE_MATH(edge, i[0], 2, i[1]-1)};
        }

        array<unsigned int, 4>
        edge2edges_ll_p1(array<unsigned int, 2> const& i) const
        {
            return array<unsigned int, 4>{
                DO_THE_MATH(edge, i[0], 0, i[1]),
                DO_THE_MATH(edge, i[0]-1, 0, i[1]+1),
                DO_THE_MATH(edge, i[0], 2, i[1]),
                DO_THE_MATH(edge, i[0]-1, 2, i[1])};
        }

        array<unsigned int, 4>
        edge2edges_ll_p2(array<unsigned int, 2> const& i) const
        {
            return array<unsigned int, 4>{
                DO_THE_MATH(edge, i[0], 0, i[1]),
                DO_THE_MATH(edge, i[0], 0, i[1]+1),
                DO_THE_MATH(edge, i[0], 1, i[1]),
                DO_THE_MATH(edge, i[0]+1, 1, i[1])};
        }

        array<unsigned int, 3>
        cell2edges_ll_p1(array<unsigned int, 2> const& i) const
        {
            return array<unsigned int, 3>{
                DO_THE_MATH(edge, i[0], 2, i[1]),
                DO_THE_MATH(edge, i[0], 0, i[1]+1),
                DO_THE_MATH(edge, i[0]+1, 1, i[1])};
        }

        array<unsigned int, 3>
        cell2edges_ll_p0(array<unsigned int, 2> const& i) const
        {
            return array<unsigned int, 3>{
                DO_THE_MATH(edge, i[0], 0, i[1]),
                DO_THE_MATH(edge, i[0], 1, i[1]),
                DO_THE_MATH(edge, i[0], 2, i[1])};
        }

        array<unsigned int, 2>
        edge2cells_ll_p0(array<unsigned int, 2> const& i) const
        {
            return array<unsigned int, 2>{
                DO_THE_MATH(cell, i[0], 1, i[1]-1),
                DO_THE_MATH(cell, i[0], 0, i[1])};
        }

        array<unsigned int, 2>
        edge2cells_ll_p1(array<unsigned int, 2> const& i) const
        {
            return array<unsigned int, 2>{
                DO_THE_MATH(cell, i[0]-1, 1, i[1]),
                DO_THE_MATH(cell, i[0], 0, i[1])};
        }

        array<unsigned int, 2>
        edge2cells_ll_p2(array<unsigned int, 2> const& i) const
        {
            return array<unsigned int, 2>{
                DO_THE_MATH(cell, i[0], 0, i[1]),
                DO_THE_MATH(cell, i[0], 1, i[1])};
        }

        array<unsigned int, 3>
        cell2cells(array<unsigned int, 2> const& i) const
        {
            if (i[1]&1) {
                return cell2cells_ll_p1({i[0], i[1]/2});
            } else {
                return cell2cells_ll_p0({i[0], i[1]/2});
            }
        }        

        array<unsigned int, 4>
        edge2edges(array<unsigned int, 2> const& i) const
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

        array<unsigned int, 3>
        cell2edges(array<unsigned int, 2> const& i) const
        {
            if (i[1]&1) {
                return cell2edges_ll_p1({i[0], i[1]/2});
            } else {
                return cell2edges_ll_p0({i[0], i[1]/2});
            }
        }

        array<unsigned int, 2>
        edge2cells(array<unsigned int, 2> const& i) const
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

