/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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
#include <vector>
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
    bool upward;
    int m_offset[3];
    static const int n_neighbors = 3;
    triangular_offsets(int, int a, bool upward)
        : upward(upward), m_offset{upward ? 1 : -1, upward ? -1 : 1, upward ? a : -a} {}

    int offset(int neighbor_index, int sign) const { return sign * m_offset[neighbor_index]; }
};

/** This is the storage class.
    The final plan would be to have generic
    stogare classes that will be parametrized
    with the structure function and some other
    functions to extract the needed information
    to compute the offsets, as for instance the
    'sign' bit in triangular_offsefs.
*/
template < typename OffsetFunction >
struct triangular_storage {
    std::vector< double > data;
    OffsetFunction offset_function;

    struct iterator : public std::iterator< std::random_access_iterator_tag, double > {
        typedef double value_type;

        std::vector< double >::iterator m_it;
        OffsetFunction const &f;
        int toggle_direction;

        iterator(std::vector< double >::iterator it, OffsetFunction const &f, int toggle_direction)
            : m_it(it), f(f), toggle_direction(toggle_direction) {}

        /** I know, using [] to access neighbors may
            seem not good, but this should highlight
            the fact that the neighbors are random-access.
        */
        double &operator[](int i) { return *(m_it + f.offset(i, toggle_direction)); }

        double const &operator[](int i) const { return *(m_it + f.offset(i, toggle_direction)); }

        double &operator*() { return *m_it; }

        iterator &operator++() {
            ++m_it;
            return *this;
        }

        iterator operator++(int) const { return m_it + 1; }

        iterator &operator--() {
            --m_it;
            return *this;
        }

        iterator operator--(int) const { return m_it - 1; }

        iterator &operator+=(int i) {
            m_it += i;
            return *this;
        }

        iterator &operator-=(int i) {
            m_it -= i;
            return *this;
        }

        iterator operator+(int i) const { return iterator(m_it + i, f, toggle_direction); }

        iterator operator-(int i) const { return iterator(m_it - i, f, toggle_direction); }

        bool operator==(iterator const &it) const { return m_it == it.m_it; }

        bool operator!=(iterator const &it) const { return m_it != it.m_it; }

        bool operator<(iterator const &it) const { return m_it < it.m_it; }

        bool operator<=(iterator const &it) const { return m_it <= it.m_it; }

        bool operator>(iterator const &it) const { return m_it > it.m_it; }

        bool operator>=(iterator const &it) const { return m_it >= it.m_it; }
    };

    triangular_storage(std::vector< double > &&data, OffsetFunction const &offset_function)
        : data(std::move(data)), offset_function(offset_function) {}

    /** NOTE THE ARGUMENT GIVEN TO BEGIN IN ORDER TO SELECT
        WHAT NEIGHBORS ARE GOING TO BE ACCESSED
    */
    iterator begin(int flag) { return iterator(data.begin(), offset_function, flag); }

    iterator end() { return iterator(data.end(), offset_function, 5); }

    /** This is the main function to perform operations
        (stencils) on grid elements. It takes a function to
        be applied. This function is the same in both the
        examples.
    */
    template < typename Functor >
    double fold_neighbors(iterator it, Functor &&f) const {
        double v = 0;
        for (int i = 0; i < OffsetFunction::n_neighbors; ++i) {
            v = f(v, it[i]);
        }
        return v;
    }
};

int main(int argc, char **argv) {
    if (argc == 1) {
        std::cout << "Usage: " << argv[0] << " n m [l] " << std::endl;
        std::cout << "Where n, m, l are the dimensions of the grid" << std::endl;
        return 0;
    }

    int n = atoi(argv[1]);
    int m = atoi(argv[2]);

    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "                          LAPLACIAN WITH TRIANGULAR MESH" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;

    {
        n = (n >> 1) << 1;
        std::cout << n << ", " << m << std::endl;

        /** Creating the storages */
        triangular_storage< triangular_offsets > storage(std::vector< double >(n * m), triangular_offsets(n, m, true));
        triangular_storage< triangular_offsets > lap(std::vector< double >(n * m), triangular_offsets(n, m, true));
        triangular_storage< triangular_offsets > lap_cool(std::vector< double >(n * m), triangular_offsets(n, m, true));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                storage.data[i * m + j] = i + j; // i+i*i*j-j;
                lap.data[i * m + j] = 0;
                lap_cool.data[i * m + j] = 0;
            }
        }

        for (int i = 0; i < n; i += std::max(1, n / 5)) {
            for (int j = 0; j < m; j += std::max(1, m / 5)) {
                std::cout << std::setw(5) << storage.data[i * m + j] << " ";
            }
            std::cout << std::endl;
        }

        /** **********************************************************
            These loops compute the Laplacian on the grid structure
            as if it was a C program
        */
        boost::timer::cpu_timer time_lap;
        for (int i = 1; i < n - 1; i += 2) {
            for (int j = 1; j < m - 1; ++j) {
                lap.data[i * m + j] =
                    3 * storage.data[i * m + j] - (storage.data[i * m + j + 1] + storage.data[i * m + j - 1] +
                                                      storage.data[i * m + j + ((j & 1) ? m : -m)]);
                // }
                // for (int j=1; j<m-1; ++j) {
                lap.data[(i + 1) * m + j] = 3 * storage.data[(i + 1) * m + j] -
                                            (storage.data[(i + 1) * m + j + 1] + storage.data[(i + 1) * m + j - 1] +
                                                storage.data[(i + 1) * m + j + ((j & 1) ? -m : m)]);
            }
        }
        boost::timer::cpu_times lapse_time_lap = time_lap.elapsed();

        std::cout << "- - - - - - - - - - - - - - - - - - - - -- -- - -- - - \n\n" << std::endl;
        for (int i = 0; i < n; i += std::max(1, n / 5)) {
            for (int j = 0; j < m; j += std::max(1, m / 5)) {
                std::cout << std::setw(5) << lap.data[i * m + j] << " ";
            }
            std::cout << std::endl;
        }

        /** **********************************************************
            These loops compute the Laplacian on the grid structure
            using the fold_neighbor function.
        */
        boost::timer::cpu_timer time_cool;
        for (int i = 1; i < n - 1; i += 2) {
            int sign = 1;
            for (int j = 1; j < m - 1; ++j) {
                lap_cool.data[i * m + j] = 3 * storage.data[i * m + j] -
                                           storage.fold_neighbors(storage.begin(sign) + i * m + j,
                                               [](double state, double value) { return state + value; });
                sign *= -1;
            }
            sign = -1;
            for (int j = 1; j < m - 1; ++j) {
                lap_cool.data[(i + 1) * m + j] = 3 * storage.data[(i + 1) * m + j] -
                                                 storage.fold_neighbors(storage.begin(sign) + (i + 1) * m + j,
                                                     [](double state, double value) { return state + value; });
                sign *= -1;
            }
        }
        boost::timer::cpu_times lapse_time_cool = time_cool.elapsed();

        std::cout << "- - - - - - - - - - - - - - - - - - - - -- -- - -- - - \n\n" << std::endl;
        for (int i = 0; i < n; i += std::max(1, n / 5)) {
            for (int j = 0; j < m; j += std::max(1, m / 5)) {
                std::cout << std::setw(5) << lap_cool.data[i * m + j] << " ";
            }
            std::cout << std::endl;
        }

        if (std::equal(lap.begin(1), lap.end(), lap_cool.begin(1))) {
            std::cout << "PASSED!" << std::endl;
        } else {
            std::cout << "FAILED!" << std::endl;
        }

        std::cout << "TIME LAP  " << boost::timer::format(lapse_time_lap) << std::endl;
        std::cout << "TIME COOL " << boost::timer::format(lapse_time_cool) << std::endl;
    }

    return 0;
}
