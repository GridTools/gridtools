#pragma once
#include "common/defs.hpp"
#include "common/array.hpp"
#include <list>
#include <vector>

namespace gridtools {
    class neighbour_list
    {
    public:
        explicit neighbour_list(array<uint_t, 4> dims)
        {
            m_neigh_indexes.resize(m_dims[0] * m_dims[1] * m_dims[2] * m_dims[3]);
            m_strides = {1,m_dims[0],m_dims[0]*m_dims[1],m_dims[0]*m_dims[1]*m_dims[2]};
            m_size = m_strides[3]*m_dims[3];
        }

        uint_t index(array<uint_t, 4> coord)
        {
            return coord*m_strides;
        }

        std::list<uint_t>& at(const uint_t i, const uint_t c, const uint_t j, const uint_t k)
        {
            return at({i,c,j,k});
        }
        std::list<uint_t>& at(array<uint_t, 4> const & coord)
        {
            assert(index(coord) < m_size);
            return m_neigh_indexes[index(coord)];
        }

        void insert_offset(array<uint_t,4> const & coord, uint_t offset)
        {
            at(coord).push_back(index(coord) + offset);
        }

    private:
        array<uint_t, 4> m_dims;
        uint_t m_size;
        std::vector<std::list<uint_t> > m_neigh_indexes;
        array<uint_t, 4> m_strides;
    };

    class unstructured_grid
    {
        static const int ncolors=2;
        explicit unstructured_grid(uint_t i, uint_t j, uint_t k) :
            m_dims{i,j,k},
            m_cell_to_cells(m_dims),
            m_cell_to_edges(m_dims),
            m_cell_to_vertexes(m_dims){}

        void construct_grid()
        {
            for(uint_t k=0; k < m_dims[2]; ++k)
            {
                for(uint_t i=1; i < m_dims[0]-1; ++i)
                {
                    for(uint_t j=1; j < m_dims[1]-1; ++j)
                    {
                        m_cell_to_cells.insert_offset({i,0,j,k}, -m_dims[0]);
                        m_cell_to_cells.insert_offset({i,0,j,k}, m_dims[0]);
                        m_cell_to_cells.insert_offset({i,0,j,k}, m_dims[0]-1);
                        m_cell_to_cells.insert_offset({i,1,j,k}, -m_dims[0]);
                        m_cell_to_cells.insert_offset({i,1,j,k}, -m_dims[0]+1);
                        m_cell_to_cells.insert_offset({i,1,j,k}, m_dims[0]);
                    }
                }
            }

        }

    private:
        array<uint_t, 3> m_dims;
        neighbour_list m_cell_to_cells;
        neighbour_list m_cell_to_edges;
        neighbour_list m_cell_to_vertexes;
    };

}//namespace gridtools
