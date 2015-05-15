#pragma once

#include <common/array.h>
#include <cassert>

namespace gridtools {

    class storage_type {
        int N,M;

    public:
        storage_type(int N, int M)
            : N{N}
            , M{M}
        {}

        unsigned int size_i() const {return N;}
        unsigned int size_j() const {return M;}

        int offset(array<unsigned int, 2> const& indices) const {
            return indices[0]*M+indices[1];
        }
    };

    template <typename ValueType>
    class cell_storage_type: public storage_type
    {
    public:
        using storage_type::storage_type;
    };

    template <typename ValueType>
    class vertex_storage_type: public storage_type
    {
    public:
        using storage_type::storage_type;
    };

    template <typename ValueType>
    class edge_storage_type: public storage_type
    {
    public:
        using storage_type::storage_type;
    };

    /**
       This class defines the maps between LocationTypes in the trapezoid grid.

       The entities are: Cells, Vertices, and Edges.

       Sizes are defined in terms of Cells.

       There are two types of maps:
       1) between index tuples
       2) between indices and offsets
    */
    template <typename T = double, typename U = double, typename V = double>
    class trapezoid_2D {
        using cell_storage_t = cell_storage_type<T>;
        using vertex_storage_t = vertex_storage_type<T>;
        using edge_storage_t = edge_storage_type<T>;

        cell_storage_t const& cell_storage;
        vertex_storage_t const& vertex_storage;
        edge_storage_t const& edge_storage;
        const unsigned int M,N; // Sizes as cells in a multi-dimensional Cell array

        static constexpr int Dims = 2;

    public:
        static constexpr unsigned int u_cell_size_j(int _M) {return _M+4;}
        static constexpr unsigned int u_cell_size_i(int _N) {return _N+2;}
        static constexpr unsigned int u_vertex_size_j(int _M) {return _M/2+1+2;}
        static constexpr unsigned int u_vertex_size_i(int _N) {return _N+1+2;}
        static constexpr unsigned int u_edge_size_j(int _M) {return 3*(_M/2)+3;}
        static constexpr unsigned int u_edge_size_i(int _N) {return _N+2;}

    private:
        int cell_size_j() const {return static_cast<int>(u_cell_size_j(M));}
        int cell_size_i() const {return static_cast<int>(u_cell_size_i(M));}
        int vertex_size_j() const {return static_cast<int>(u_vertex_size_j(M));}
        int vertex_size_i() const {return static_cast<int>(u_vertex_size_i(M));}
        int edge_size_j() const {return static_cast<int>(u_edge_size_j(M));}
        int edge_size_i() const {return static_cast<int>(u_edge_size_i(M));}

        unsigned int u_cell_size_j() const {return (u_cell_size_j(M));}
        unsigned int u_cell_size_i() const {return (u_cell_size_i(M));}
        unsigned int u_vertex_size_j() const {return (u_vertex_size_j(M));}
        unsigned int u_vertex_size_i() const {return (u_vertex_size_i(M));}
        unsigned int u_edge_size_j() const {return (u_edge_size_j(M));}
        unsigned int u_edge_size_i() const {return (u_edge_size_i(M));}

    public :
        trapezoid_2D(cell_storage_t const& cs, vertex_storage_t const& vs, edge_storage_t const& es)
            : cell_storage(cs)
            , vertex_storage(vs)
            , edge_storage(es)
            , M{cell_storage.size_j()}
            , N{cell_storage.size_i()}
        {
            assert((M&1) == 0);
            assert((N&1) == 0);
        }

        array<array<unsigned int, Dims>, 3> cell2cells(array<unsigned int, Dims> const& indices) const {
            return array<array<unsigned int, Dims>, 3>{array<unsigned int, Dims>{indices[0], indices[1]-1},
                                                       array<unsigned int, Dims>{indices[0], indices[1]+1},
                                                       array<unsigned int, Dims>{(indices[1]&1)?indices[0]+1:indices[0]-1, indices[1]+((indices[1]&1)?-1:1)}};
            }
        array<int, 3> cell2cells_offsets(array<unsigned int, Dims> const& indices) const {
            return array<int, 3>{-1, 1, (indices[1]&1)?cell_size_j():-cell_size_j()};
            }
        array<array<unsigned int, Dims>, 6> vertex2vertices(array<unsigned int, Dims> const& indices) const {
            return array<array<unsigned int, Dims>, 6>{ array<unsigned int, Dims>{indices[0]-1, indices[1]},
                                                        array<unsigned int, Dims>{indices[0]-1, indices[1]+1},
                                                        array<unsigned int, Dims>{indices[0], indices[1]-2},
                                                        array<unsigned int, Dims>{indices[0], indices[1]+1},
                                                        array<unsigned int, Dims>{indices[0]+1, indices[1]-1},
                                                        array<unsigned int, Dims>{indices[0]+1, indices[1]}
                                                      };
        }

        array<int, 6> vertex2vertices_offsets(array<unsigned int, Dims> const&) const {
            return array<int, 6>{-vertex_size_j(), -vertex_size_j()+1, -1, 1, vertex_size_j()-1, vertex_size_j()};
        }
        array<array<unsigned int, Dims>, 4> edge2edges(array<unsigned int, Dims> const& indices) const {
            return array<array<unsigned int, Dims>, 4>{ {indices[0], indices[1]-1}, 
                                                        {indices[0], indices[1]+1},
                                                        {indices[0] + ((indices[1]&1)?1:-1), indices[1]+ ((indices[1]&1)?2:-2)},
                                                        {indices[0]+1, indices[1]+((indices[1]&1)?-2:2)}
                                                      };
        }

        array<int, 4> edge2edges_offsets(array<unsigned int, Dims> const& indices) const {
            return array<int, 4>{ -1, +1, (indices[1]&1)?-2:2, edge_size_j()+ ((indices[1]&1)?0:1)};
        }

        array<array<unsigned int, Dims>, 3> cell2vertices(array<unsigned int, Dims> const& indices) const {
            return array<array<unsigned int, Dims>, 3>{ {indices[0], indices[1]}, 
                                                        {(indices[0]&1)?indices[0]:(indices[0]+1), indices[1]+ ((indices[0]&1)?1:-1)}, 
                                                        {indices[0]+1, indices[1]}
                                                      };
        }
        array<int, 3> cell2vertices_offsets(array<unsigned int, Dims> const& indices) const {
            return array<int, 3>{ edge_storage.offset(indices), // no need for computing anything here, sice {indices[0], indices[1]} is also the index of the reference vertex in the neighbohrood of the node
                                  (indices[0]&1)?vertex_size_j():1,
                                  (indices[0]&1)?vertex_size_j()+1:vertex_size_j()
                                };
        }

        array<array<unsigned int, Dims>, 3> cell2edges(array<unsigned int, Dims> const& indices) const {
            return array<array<unsigned int, Dims>, 3>{ {indices[0] + ((indices[0]&1)?1:0), (indices[1]/2)*3+1},
                                                        {indices[0], (indices[1]/2)*3+2},
                                                        {indices[0], indices[1] + ((indices[0]&1)?3:0)}
                                                      };
        }
        array<int, 3> cell2edges_offsets(array<unsigned int, Dims> const& indices) const {
            return array<int, 3>{edge_storage.offset({indices[0], (indices[1]/2)+1}), 1, (indices[0]&1)?3:0};
        }

//         array<array<unsigned int, Dims>, 6> vertex2cells(array<unsigned int, Dims> const&) const {
//         }
//         array<array<unsigned int, Dims>, 6> vertex2edges(array<unsigned int, Dims> const&) const {
//         }
//         array<array<unsigned int, Dims>, 2> edge2cells(array<unsigned int, Dims> const&) const {
//         }
//         array<array<unsigned int, Dims>, 2> edge2verices(array<unsigned int, Dims> const&) const {
//         }

    };

}

