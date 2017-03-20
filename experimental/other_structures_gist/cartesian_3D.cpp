/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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
#include <cmath>
#include <boost/timer/timer.hpppp>
#include <thread>
#include <cstdlib>

/** This is the function which defines the structure
    i.e., define the offsets of a node.
    In this case the function does not need any
    parameter since the offsets do not depend on
    the current node. The offsets depend on the
    dimensions of the grid.
*/
struct neighbor_offsets {
    int m_offset[6];
    static const int n_neighbors = 6;
    neighbor_offsets(int, int b, int c) : m_offset{-b * c, b * c, -c, c, -1, 1} {
        // int i=0;
        // std::for_each(&m_offset[0], &m_offset[6], [&i](int x) {std::cout << "Offset " << i++ << " " << x <<
        // std::endl;});
    }

    neighbor_offsets() : m_offset{0, 0, 0, 0, 0, 0} {}

    int offset(int neighbor_index) const { return m_offset[neighbor_index]; }
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
struct structured_storage {
    std::vector< double > data;
    OffsetFunction offset_function;

    struct iterator : public std::iterator< std::random_access_iterator_tag, double > {
        typedef double value_type;

        std::vector< double >::iterator m_it;
        OffsetFunction const &f;

        iterator(std::vector< double >::iterator it, OffsetFunction const &f) : m_it(it), f(f) {}

        /** I know, using [] to access neighbors may
            seem not good, but this should highlight
            the fact that the neighbors are random-access.
        */
        double &operator[](int i) { return *(m_it + f.offset(i)); }

        double const &operator[](int i) const { return *(m_it + f.offset(i)); }

        double &operator*() { return *m_it; }

        double const &operator*() const { return *m_it; }

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

        iterator operator+(int i) const { return iterator(m_it + i, f); }

        iterator operator-(int i) const { return iterator(m_it - i, f); }

        bool operator==(iterator const &it) const { return m_it == it.m_it; }

        bool operator!=(iterator const &it) const { return m_it != it.m_it; }

        bool operator<(iterator const &it) const { return m_it < it.m_it; }

        bool operator<=(iterator const &it) const { return m_it <= it.m_it; }

        bool operator>(iterator const &it) const { return m_it > it.m_it; }

        bool operator>=(iterator const &it) const { return m_it >= it.m_it; }
    };

    structured_storage(std::vector< double > &&data, OffsetFunction const &offset_function)
        : data(std::move(data)), offset_function(offset_function) {}

    structured_storage() : data(), offset_function() {}

    iterator begin() { return iterator(data.begin(), offset_function); }

    iterator end() { return iterator(data.end(), offset_function); }

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

#ifndef INCLUDE_ONLY
int main(int argc, char **argv) {
    if (argc == 1) {
        std::cout << "Usage: " << argv[0] << " n m l " << std::endl;
        std::cout << "Where n, m, l are the dimensions of the grid" << std::endl;
        return 0;
    }
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int l = atoi(argv[3]);

    int n_threads = 1;

    if (const char *omp_threads = std::getenv("OMP_NUM_THREADS")) {
        n_threads = std::atoi(omp_threads);
        std::cout << "Number of threads have been set to " << n_threads << std::endl;
    }

    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "                            REGULAR LAPLACIAN IN 3D" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;
    std::cout << "  *****************************************************************************" << std::endl;
    {
        std::cout << n << ", " << m << ", " << l << std::endl;

        /** Creating the storages */
        structured_storage< neighbor_offsets > storage(std::vector< double >(n * m * l), neighbor_offsets(n, m, l));
        structured_storage< neighbor_offsets > lap(std::vector< double >(n * m * l), neighbor_offsets(n, m, l));
        structured_storage< neighbor_offsets > lap_cool(std::vector< double >(n * m * l), neighbor_offsets(n, m, l));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                for (int k = 0; k < l; ++k) {
                    storage.data[i * m * l + j * l + k] = static_cast< double >(i * k * k + i * i * k * j - k * i) /
                                                          static_cast< double >(n * l * l + n * n * l * m - l * n);
                    lap.data[i * m * l + j * l + k] = 0;
                    lap_cool.data[i * m * l + j * l + k] = 0;
                }
            }
        }

        for (int i = 0; i < n; i += std::max(1, n / 5)) {
            for (int j = 0; j < m; j += std::max(1, m / 5)) {
                for (int k = 0; k < l; k += std::max(1, l / 10)) {
                    std::cout << std::setw(5) << storage.data[i * m * l + j * l + k] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        // is it better to use constexpr here or static const? It does not impact performance.
        constexpr double iminus = 1;
        constexpr double iplus = -4;
        constexpr double jminus = 2;
        constexpr double jplus = -1;
        constexpr double kminus = 5;
        constexpr double kplus = 3;
        // constexpr in the next is not really needed
        constexpr double coeff[6] = {iminus, iplus, jminus, jplus, kminus, kplus};
        // Using std::vector is more than 30% slower
        // std::vector<double> coeff = {iminus, iplus, jminus, jplus, kminus, kplus};

        std::vector< std::thread > threads(n_threads);

        /** **********************************************************
            These loops compute the Laplacian on the grid structure
            as if it was a C program
        */
        int iter_space = (n - 2) / n_threads;
        std::vector< std::string > th_output(n_threads);
        boost::timer::cpu_timer time_lap;
        for (int th = 0; th < n_threads; ++th) {
            threads[th] = std::thread([th, n, m, l, iter_space, &lap, &storage, &th_output] {
                th_output[th] = std::to_string(1 + (iter_space)*th) + " to " + std::to_string((iter_space) * (th + 1)) +
                                " n=" + std::to_string(n) + " m=" + std::to_string(m) + " l=" + std::to_string(l);
                for (int i = 1 + (iter_space)*th; i <= (iter_space) * (th + 1); ++i) {
                    for (int j = 1; j < m - 1; ++j) {
                        for (int k = 1; k < l - 1; ++k) {
                            lap.data[i * m * l + j * l + k] = 6 * storage.data[i * m * l + j * l + k] -
                                                              (storage.data[(i + 1) * m * l + j * l + k] * iplus +
                                                                  storage.data[(i - 1) * m * l + j * l + k] * iminus +
                                                                  storage.data[i * m * l + (j + 1) * l + k] * jplus +
                                                                  storage.data[i * m * l + (j - 1) * l + k] * jminus +
                                                                  storage.data[i * m * l + j * l + k + 1] * kplus +
                                                                  storage.data[i * m * l + j * l + k - 1] * kminus);
                        }
                    }
                }
            });
        }
        for (int th = 0; th < n_threads; ++th) {
            threads[th].join();
        }

        boost::timer::cpu_times lapse_time_lap = time_lap.elapsed();

        for (int th = 0; th < n_threads; ++th) {
            std::cout << "Output from thread " << th << ": " << th_output[th] << std::endl;
        }

        std::cout << "- - - - - - - - - - - - - - - - - - - - -- -- - -- - - \n\n" << std::endl;
        for (int i = 0; i < n; i += std::max(1, n / 5)) {
            for (int j = 0; j < m; j += std::max(1, m / 5)) {
                for (int k = 0; k < l; k += std::max(1, l / 10)) {
                    std::cout << std::setw(5) << lap.data[i * m * l + j * l + k] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        /** **********************************************************
            These loops compute the Laplacian on the grid structure
            using the fold_neighbor function.
        */
        boost::timer::cpu_timer time_cool;
        //#pragma omp parallel for collapse(3)
        for (int th = 0; th < n_threads; ++th) {
            threads[th] = std::thread([th, n, m, l, iter_space, &lap_cool, &storage, coeff /*,&th_output*/] {
                for (int i = 1 + (iter_space)*th; i <= (iter_space) * (th + 1); ++i) {
                    for (int j = 1; j < m - 1; ++j) {
                        for (int k = 1; k < l - 1; ++k) {
                            int neigh = 0;
                            lap_cool.data[i * m * l + j * l + k] =
                                6 * storage.data[i * m * l + j * l + k] -
                                storage.fold_neighbors(storage.begin() + i * m * l + j * l + k,
                                    [&coeff, &neigh](double const state, double const value) {
                                        return state + value * coeff[neigh++];
                                    });
                        }
                    }
                }
            });
        }
        for (int th = 0; th < n_threads; ++th) {
            threads[th].join();
        }

        boost::timer::cpu_times lapse_time_cool = time_cool.elapsed();

        std::cout << "- - - - - - - - - - - - - - - - - - - - -- -- - -- - - \n\n" << std::endl;
        for (int i = 0; i < n; i += std::max(1, n / 5)) {
            for (int j = 0; j < m; j += std::max(1, m / 5)) {
                for (int k = 0; k < l; k += std::max(1, l / 10)) {
                    std::cout << std::setw(5) << lap_cool.data[i * m * l + j * l + k] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        if (std::equal(lap.begin(),
                lap.end(),
                lap_cool.begin(),
                [](double const &v1, double const &v2) -> bool { /* if (std::abs((v1)-(v2))>=1e-10)
                            {std::cout << std::abs((v1)-(v2)) << std::endl;} */
                    return (std::abs((v1) - (v2)) < 1e-10);
                })) {
            std::cout << "PASSED!" << std::endl;
        } else {
            std::cout << "FAILED!" << std::endl;
        }

        std::cout << "TIME LAP  " << boost::timer::format(lapse_time_lap) << std::endl;
        std::cout << "TIME COOL " << boost::timer::format(lapse_time_cool) << std::endl;
    }

    return 0;
}
#endif
