#pragma once

#include <vector>
#include <list>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <stdlib.h>
#include <boost/timer/timer.hpp>


/** This is the function which defines the structure
    i.e., define the offsets of a node.
    In this case the function needs a parameter,
    a bit, to determine if to move to the left or
    to the right. The offsets depend on the dimensions
    of the grid.
*/
struct triangular_offsets {
    int m_offset[3];
    static const int n_neighbors = 3;
    triangular_offsets(int a) {
        m_offset[0] = 1;
        m_offset[1] = -1;
        m_offset[2] = a;
    }

    int offset(int neighbor_index, int sign) const {
        return sign*m_offset[neighbor_index];
    }

};


/** This is the storage class.
    The final plan would be to have generic
    stogare classes that will be parametrized
    with the structure function and some other
    functions to extract the needed information
    to compute the offsets, as for instance the
    'sign' bit in triangular_offsefs.
*/
template <typename OffsetFunction>
struct triangular_storage {
    std::vector<double> data;
    OffsetFunction offset_function;
    int nRows, nColumns, haloSize;

    struct iterator : public std::iterator<std::random_access_iterator_tag, double> {
        typedef double value_type;

        std::vector<double>::iterator m_it;
        OffsetFunction const& f;
        int toggle_direction;

        iterator(std::vector<double>::iterator it, OffsetFunction const& f, int toggle_direction)
        : m_it(it)
        , f(f)
        , toggle_direction(toggle_direction)
        { }

        /** I know, using [] to access neighbors may
            seem not good, but this should highlight
            the fact that the neighbors are random-access.
        */
        double& operator[](int i) {
            return *(m_it+f.offset(i, toggle_direction));
        }

        double const& operator[](int i) const {
            return *(m_it+f.offset(i, toggle_direction));
        }

        double& operator*() {
            return *m_it;
        }

        iterator & operator++() {
            ++m_it;
            toggle_direction = toggle_direction*(-1);
            return *this;
        }

        iterator operator++(int) const {
            toggle_direction = toggle_direction*(-1);
            return m_it+1;
        }

        iterator & operator--() {
            --m_it;
            toggle_direction = toggle_direction*(-1);
            return *this;
        }

        iterator operator--(int) const {
            toggle_direction = toggle_direction*(-1);
            return m_it-1;
        }

        iterator & operator+=(int i) {
            m_it += i;
            toggle_direction = toggle_direction*(i&1 ? -1 : 1);
            return *this;
        }

        iterator & operator-=(int i) {
            m_it -= i;
            toggle_direction = toggle_direction*(i&1 ? -1 : 1);
            return *this;
        }

        iterator operator+(int i) const {
            return iterator(m_it+i, f, toggle_direction*(i&1 ? -1 : 1));
        }

        iterator operator-(int i) const {
            return iterator(m_it-i, f, toggle_direction*(i&1 ? -1 : 1));
        }

        bool operator==(iterator const& it) const {
            return m_it==it.m_it;
        }

        bool operator!=(iterator const& it) const {
            return m_it!=it.m_it;
        }

        bool operator<(iterator const& it) const {
            return m_it<it.m_it;
        }

        bool operator<=(iterator const& it) const {
            return m_it<=it.m_it;
        }

        bool operator>(iterator const& it) const {
            return m_it>it.m_it;
        }

        bool operator>=(iterator const& it) const {
            return m_it>=it.m_it;
        }
        iterator neighbor(int i) {
            assert(i>=0 && i < 3);
            return iterator(m_it+f.offset(i, toggle_direction), f, toggle_direction*(-1));
        }
        int neighbor_off(int i) {
            assert(i>=0 && i < 3);
            return f.offset(i, toggle_direction);
        }
    };

    triangular_storage(std::vector<double> && data, int nrows, int ncolumns, int halosize, OffsetFunction const & offset_function)
        : data(std::move(data))
        , offset_function(offset_function), nRows(nrows), nColumns(ncolumns), haloSize(halosize)
    { }

    /** NOTE THE ARGUMENT GIVEN TO BEGIN IN ORDER TO SELECT
        WHAT NEIGHBORS ARE GOING TO BE ACCESSED
    */
    iterator begin() {
        return iterator(data.begin(), offset_function, 1);
    }

    iterator end() {
        return iterator(data.end(), offset_function, 5);
    }

    /** This is the main function to perform operations
        (stencils) on grid elements. It takes a function to
        be applied. This function is the same in both the
        examples.
    */
    template <typename Functor>
    double fold_neighbors(iterator it, Functor && f) const {
        double v = 0;
        for (int i=0; i<OffsetFunction::n_neighbors; ++i) {
            v = f(v, it[i]);
        }
        return v;
    }

    template <typename Functor>
    double fold_neighbors_dbg(iterator it, double center, Functor && f) const {
        double v = 3*center;
        std::cout << " c@ " << v;
        for (int i=0; i<OffsetFunction::n_neighbors; ++i) {
            std::cout << " f@ " << it.neighbor_off(i) << " @v " << it[i];
            v = f(v, it[i]);
        }
        std::cout << " v " << v;
        return v;
    }


    template <typename Functor>
    double fold_2nd_neighbors(iterator it, double center, Functor && f) const {
        double v = center;
        std::cout << " @2c " << center; //1.5
        for (int i=0; i<OffsetFunction::n_neighbors; ++i) {
            std::cout << " n@ " << it.neighbor_off(i);
            v = f(v, fold_neighbors_dbg(it.neighbor(i), *it.neighbor(i), f));
        }//-3
        std::cout << " @2t " << v;
        return v;
    }

    int StartComputationDomain() const
    {
        // n cells in halo of diamond 3
        int pos = pow(haloSize,2)*2;
        // add cells from halo of diamond 2
        pos += nColumns*haloSize*2;
        // add cells from padding between diamonds 2 and 6
        pos += (haloSize-1)*haloSize*2+1;
        // add cells of one halo row in diamonds 4 and 5
        pos += haloSize*2;
        // return position of first cell in compuation domain
        return pos;
    }
    int EndComputationDomain() const
    {
        // n cells in halo of diamond 3
        int pos = pow(haloSize,2)*2;
        // add cells from halo of diamond 2
        pos += nColumns*haloSize*2;
        // add cells from padding between diamonds 2 and 6
        pos += (haloSize-1)*haloSize*2+1;
        // add all cells in computation domain plus halos in diamonds 4-5/ and 6
        pos+= (nColumns+haloSize*2)*nRows*2;
        // substract the last row in halo in diamond 6
        pos -= haloSize*2;
        return pos;
    }
    int RowId(int cellId) const
    {
        int stride = (haloSize*2+nColumns)*2;
        return (cellId - (StartComputationDomain() - haloSize*2))/stride;
    }

    int ColumnId(int cellId) const
    {
        int stride = (haloSize*2+nColumns)*2;
        int columnId = ((cellId - (StartComputationDomain() - haloSize*2))%stride - haloSize*2);
        return columnId < 0 ? (columnId)/2 -1 : columnId/2;
    }

    bool CellInComputeDomain(int cellId) const
    {
        int rowId = RowId(cellId);
        int columnId = ColumnId(cellId);
        return cellId >= StartComputationDomain() &&
                cellId < EndComputationDomain() &&
                rowId >= 0 && rowId < nRows &&
                columnId >= 0 && columnId < nColumns;
    }

};
