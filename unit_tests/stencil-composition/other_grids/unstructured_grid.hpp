#pragma once
#include "common/defs.hpp"
#include "common/array.hpp"
#include <list>
#include <vector>
#include <iostream>

namespace gridtools {
    class neighbour_list
    {
    public:
        explicit neighbour_list(array<uint_t, 4>& dims)
        {
            m_neigh_indexes.resize(dims[0] * dims[1] * dims[2] * dims[3]);
            m_strides = {1,dims[0],dims[0]*dims[1],dims[0]*dims[1]*dims[2]};
            m_size = m_strides[3]*dims[3];
        }

        uint_t index(array<uint_t, 4> coord)
        {
            return coord*m_strides;
        }

        std::list<array<uint_t,4> >& at(const uint_t i, const uint_t c, const uint_t j, const uint_t k)
        {
            return at({i,c,j,k});
        }
        std::list<array<uint_t,4> >& at(array<uint_t, 4> const & coord)
        {
            assert(index(coord) < m_size);
            return m_neigh_indexes[index(coord)];
        }

        void insert_neighbour(array<uint_t,4> const & coord, array<uint_t, 4> neighbour)
        {
            at(coord).push_back(neighbour);
        }

    private:
        uint_t m_size;
        std::vector<std::list<array<uint_t, 4> > > m_neigh_indexes;
        array<uint_t, 4> m_strides;
    };

    class unstructured_grid
    {
        static const int ncolors=2;
    public:
        explicit unstructured_grid(uint_t i, uint_t j, uint_t k) :
            m_dims{i,2,j,k},
            m_cell_to_cells(m_dims),
            m_cell_to_edges(m_dims),
            m_cell_to_vertexes(m_dims)
        {
            construct_grid();
        }

        void construct_grid()
        {
            for(uint_t k=0; k < m_dims[3]; ++k)
            {
                for(uint_t i=1; i < m_dims[0]-1; ++i)
                {
                    for(uint_t j=1; j < m_dims[2]-1; ++j)
                    {
                        m_cell_to_cells.insert_neighbour({i,0,j,k}, {i,1,j-1,k});
                        m_cell_to_cells.insert_neighbour({i,0,j,k}, {i-1,1,j,k});
                        m_cell_to_cells.insert_neighbour({i,0,j,k}, {i,1,j,k});
                        m_cell_to_cells.insert_neighbour({i,1,j,k}, {i,0,j,k});
                        m_cell_to_cells.insert_neighbour({i,1,j,k}, {i+1,0,j,k});
                        m_cell_to_cells.insert_neighbour({i,1,j,k}, {i,0,j+1,k});
                    }
                }
            }
        }

        std::list<array<uint_t, 4>> const& neighbours_of(array<uint_t, 4> const& coords)
        {
            return m_cell_to_cells.at(coords);
        }

    private:
        array<uint_t, 4> m_dims;
        neighbour_list m_cell_to_cells;
        neighbour_list m_cell_to_edges;
        neighbour_list m_cell_to_vertexes;
    };

}//namespace gridtools
