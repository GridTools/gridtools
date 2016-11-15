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
#include <iostream>
#include <fstream>
#include <gridtools.hpp>
#include <common/defs.hpp>
#include <ctime>

#define offs_(i, j, k, n, m, l) ((i) * (m) * (l) + (j) * (l) + (k))

using gridtools::uint_t;
using gridtools::int_t;

template < typename Stream >
void print(double *that, uint_t n, uint_t m, uint_t l, Stream &stream) {
    // std::cout << "Printing " << name << std::endl;
    stream << "(" << n << "x" << m << "x" << l << ")" << std::endl;
    stream << "| j" << std::endl;
    stream << "| j" << std::endl;
    stream << "v j" << std::endl;
    stream << "---> k" << std::endl;

    uint_t MI = 12;
    uint_t MJ = 12;
    uint_t MK = 12;

    for (uint_t i = 0; i < n; i += std::max((uint_t)1, n / MI)) {
        for (uint_t j = 0; j < m; j += std::max((uint_t)1, m / MJ)) {
            for (uint_t k = 0; k < l; k += std::max((uint_t)1, l / MK)) {
                stream << "[" /*("
                                  << i << ","
                                  << j << ","
                                  << k << ")"*/
                       << that[offs_(i, j, k, n, m, l)] << "] ";
            }
            stream << std::endl;
        }
        stream << std::endl;
    }
    stream << std::endl;
}

int main_naive(int argc, char **argv) {

    if (argc != 4) {
        std::cout << "Usage: basic_laplacian dimx dimy dimz\n where args are integer sizes of the data fields"
                  << std::endl;
        return 1;
    }

    /**
       The following steps are performed:

       - Definition of the domain:
    */
    uint_t d1 = atoi(argv[1]); /** d1 cells in the x direction (horizontal)*/
    uint_t d2 = atoi(argv[2]); /** d2 cells in the y direction (horizontal)*/
    uint_t d3 = atoi(argv[3]); /** d3 cells in the z direction (vertical)*/

    std::ofstream file_i("vanilla_naive_in");
    std::ofstream file_o("vanilla_naive_out");

    double *in = new double[d1 * d2 * d3];
    double *out = new double[d1 * d2 * d3];

    srand(12345);
    // DO NOT FUSE THESE LOOPS!!!
    for (uint_t i = 0; i < d1 * d2 * d3; ++i) {
        in[i] = -1.0 * rand();
    }

    srand(12345);
    for (uint_t i = 0; i < d1 * d2 * d3; ++i) {
        out[i] = -7.3 * rand();
    }

    print(out, d1, d2, d3, file_i);

    for (uint_t i = 2; i < d1 - 2; ++i) {
        for (uint_t j = 2; j < d2 - 2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                // std::cout << in(i,j,k) << std::endl;
                assert(offs_(i, j, k, d1, d2, d3) >= 0);
                assert(offs_(i, j, k, d1, d2, d3) < d1 * d2 * d3);
                // std::cout << i << ", " << j << ", " << k << " - " << d1 << ", " << d2 << ", " << d3 << " -- " <<
                // offs_(i,j,k,d1,d2,d3) << " " << d1*d2*d3 << std::endl;
                // std::cout << i << ", " << j << ", " << k << " - " << d1 << ", " << d2 << ", " << d3 << " -- " <<
                // offs_(i+1,j,k,d1,d2,d3) << " " << d1*d2*d3 << std::endl;
                // std::cout << i << ", " << j << ", " << k << " - " << d1 << ", " << d2 << ", " << d3 << " -- " <<
                // offs_(i-1,j,k,d1,d2,d3) << " " << d1*d2*d3 << std::endl;
                out[offs_(i, j, k, d1, d2, d3)] =
                    4 * in[offs_(i, j, k, d1, d2, d3)] -
                    (in[offs_(i + 1, j, k, d1, d2, d3)] + in[offs_(i, j + 1, k, d1, d2, d3)] +
                        in[offs_(i - 1, j, k, d1, d2, d3)] + in[offs_(i, j - 1, k, d1, d2, d3)]);
            }
        }
    }
    print(out, d1, d2, d3, file_o);

    delete[] in;
    delete[] out;

    return 0;
}

int main_naive_inc(int argc, char **argv) {

    if (argc != 4) {
        std::cout << "Usage: basic_laplacian dimx dimy dimz\n where args are integer sizes of the data fields"
                  << std::endl;
        return 1;
    }

    /**
       The following steps are performed:

       - Definition of the domain:
    */
    uint_t d1 = atoi(argv[1]); /** d1 cells in the x direction (horizontal)*/
    uint_t d2 = atoi(argv[2]); /** d2 cells in the y direction (horizontal)*/
    uint_t d3 = atoi(argv[3]); /** d3 cells in the z direction (vertical)*/

    std::ofstream file_i("vanilla_naive_inc_in");
    std::ofstream file_o("vanilla_naive_inc_out");

    double *in = new double[d1 * d2 * d3];
    double *out = new double[d1 * d2 * d3];

    srand(12345);
    // DO NOT FUSE THESE LOOPS!!!
    for (uint_t i = 0; i < d1 * d2 * d3; ++i) {
        in[i] = -1.0 * rand();
    }

    srand(12345);
    for (uint_t i = 0; i < d1 * d2 * d3; ++i) {
        out[i] = -7.3 * rand();
    }

    print(out, d1, d2, d3, file_i);

    std::clock_t start;
    long double duration;
    start = std::clock();

    for (uint_t i = 2; i < d1 - 2; ++i) {
        for (uint_t j = 2; j < d2 - 2; ++j) {
            double *po = out + offs_(i, j, 0, d1, d2, d3);
            double *pi0 = in + offs_(i, j, 0, d1, d2, d3);
            double *pi1 = in + offs_(i + 1, j, 0, d1, d2, d3);
            double *pi2 = in + offs_(i, j + 1, 0, d1, d2, d3);
            double *pi3 = in + offs_(i - 1, j, 0, d1, d2, d3);
            double *pi4 = in + offs_(i, j - 1, 0, d1, d2, d3);
            for (uint_t k = 0; k < d3; ++k) {
                // std::cout << in(i,j,k) << std::endl;
                assert(offs_(i, j, k, d1, d2, d3) >= 0);
                assert(offs_(i, j, k, d1, d2, d3) < d1 * d2 * d3);
                // std::cout << i << ", " << j << ", " << k << " - " << d1 << ", " << d2 << ", " << d3 << " -- " <<
                // offs_(i,j,k,d1,d2,d3) << " " << d1*d2*d3 << std::endl;
                // std::cout << i << ", " << j << ", " << k << " - " << d1 << ", " << d2 << ", " << d3 << " -- " <<
                // offs_(i+1,j,k,d1,d2,d3) << " " << d1*d2*d3 << std::endl;
                // std::cout << i << ", " << j << ", " << k << " - " << d1 << ", " << d2 << ", " << d3 << " -- " <<
                // offs_(i-1,j,k,d1,d2,d3) << " " << d1*d2*d3 << std::endl;
                *po = 4 * *pi0 - (*pi1 + *pi2 + *pi3 + *pi4);
                ++po;
                ++pi0;
                ++pi1;
                ++pi2;
                ++pi3;
                ++pi4;
                // out(i,j,k) = 4 * in(i,j,k) -
                //     (in( i+1, j, k) + in( i, j+1, k) +
                //      in( i-1, j, k) + in( i, j-1, k));
            }
        }
    }
    duration = (std::clock() - start) / (long double)CLOCKS_PER_SEC;

    print(out, d1, d2, d3, file_o);

    std::cout << "TIME " << duration << std::endl;

    delete[] in;
    delete[] out;

    return 0;
}

int main_block(int argc, char **argv) {

    if (argc != 4) {
        std::cout << "Usage: basic_laplacian dimx dimy dimz\n where args are integer sizes of the data fields"
                  << std::endl;
        return 1;
    }

    /**
       The following steps are performed:

       - Definition of the domain:
    */
    uint_t d1 = atoi(argv[1]); /** d1 cells in the x direction (horizontal)*/
    uint_t d2 = atoi(argv[2]); /** d2 cells in the y direction (horizontal)*/
    uint_t d3 = atoi(argv[3]); /** d3 cells in the z direction (vertical)*/

    std::ofstream file_i("vanilla_block_in");
    std::ofstream file_o("vanilla_block_out");

    double *in = new double[d1 * d2 * d3];
    double *out = new double[d1 * d2 * d3];

    srand(12345);
    // DO NOT FUSE THESE LOOPS!!!
    for (uint_t i = 0; i < d1 * d2 * d3; ++i) {
        in[i] = -1.0 * rand();
    }

    srand(12345);
    for (uint_t i = 0; i < d1 * d2 * d3; ++i) {
        out[i] = -7.3 * rand();
    }

    print(out, d1, d2, d3, file_i);

    std::clock_t start;
    long double duration;
    start = std::clock();

    uint_t BI = 4;
    uint_t BJ = 4;

    uint_t NBI = (d1 - 4) / BI;
    uint_t NBJ = (d2 - 4) / BJ;
    {
        for (uint_t bi = 0; bi < NBI; ++bi) {
            for (uint_t bj = 0; bj < NBJ; ++bj) {
                uint_t starti = bi * BI + 2;
                uint_t startj = bj * BJ + 2;
                for (uint_t i = starti; i < starti + BI; ++i) {
                    for (uint_t j = startj; j < startj + BJ; ++j) {
#ifndef NDEBUG
                        std::cout << "B1"
                                  << " "
                                  << "starti " << starti << " "
                                  << " i " << i << " end " << starti + BI << "\n   startj " << startj << " j " << j
                                  << " end " << startj + BJ << std::endl;
#endif
                        for (uint_t k = 0; k < d3; ++k) {
                            assert(offs_(i, j, k, d1, d2, d3) >= 0);
                            assert(offs_(i, j, k, d1, d2, d3) < d1 * d2 * d3);
                            out[offs_(i, j, k, d1, d2, d3)] =
                                4 * in[offs_(i, j, k, d1, d2, d3)] -
                                (in[offs_(i + 1, j, k, d1, d2, d3)] + in[offs_(i, j + 1, k, d1, d2, d3)] +
                                    in[offs_(i - 1, j, k, d1, d2, d3)] + in[offs_(i, j - 1, k, d1, d2, d3)]);
                        }
                    }
                }
            }
        }

        for (uint_t bj = 0; bj < NBJ; ++bj) {
            uint_t starti = NBI * BI + 2;
            uint_t startj = bj * BJ + 2;
            for (uint_t i = starti; i < d1 - 2; ++i) {
                for (uint_t j = startj; j < startj + BJ; ++j) {
#ifndef NDEBUG
                    std::cout << "B2"
                              << " "
                              << "starti " << starti << " "
                              << " i " << i << " end " << d1 - 2 << "\n   startj " << startj << " j " << j << " end "
                              << startj + BJ << std::endl;
#endif
                    for (uint_t k = 0; k < d3; ++k) {
                        assert(offs_(i, j, k, d1, d2, d3) >= 0);
                        assert(offs_(i, j, k, d1, d2, d3) < d1 * d2 * d3);
                        out[offs_(i, j, k, d1, d2, d3)] =
                            4 * in[offs_(i, j, k, d1, d2, d3)] -
                            (in[offs_(i + 1, j, k, d1, d2, d3)] + in[offs_(i, j + 1, k, d1, d2, d3)] +
                                in[offs_(i - 1, j, k, d1, d2, d3)] + in[offs_(i, j - 1, k, d1, d2, d3)]);
                    }
                }
            }
        }

        for (uint_t bi = 0; bi < NBI; ++bi) {
            uint_t starti = bi * BI + 2;
            uint_t startj = NBJ * BJ + 2;
            for (uint_t i = starti; i < starti + BI; ++i) {
                for (uint_t j = startj; j < d2 - 2; ++j) {
#ifndef NDEBUG
                    std::cout << "B3"
                              << " "
                              << "starti " << starti << " "
                              << " i " << i << " end " << starti + BI << "\n   startj " << startj << " j " << j
                              << " end " << d2 - 2 << std::endl;
#endif
                    for (uint_t k = 0; k < d3; ++k) {
                        assert(offs_(i, j, k, d1, d2, d3) >= 0);
                        assert(offs_(i, j, k, d1, d2, d3) < d1 * d2 * d3);
                        out[offs_(i, j, k, d1, d2, d3)] =
                            4 * in[offs_(i, j, k, d1, d2, d3)] -
                            (in[offs_(i + 1, j, k, d1, d2, d3)] + in[offs_(i, j + 1, k, d1, d2, d3)] +
                                in[offs_(i - 1, j, k, d1, d2, d3)] + in[offs_(i, j - 1, k, d1, d2, d3)]);
                    }
                }
            }
        }

        uint_t starti = NBI * BI + 2;
        uint_t startj = NBJ * BJ + 2;
        for (uint_t i = starti; i < d1 - 2; ++i) {
            for (uint_t j = startj; j < d2 - 2; ++j) {
#ifndef NDEBUG
                std::cout << "B4"
                          << " "
                          << "starti " << starti << " "
                          << " i " << i << " end " << d1 - 2 << "\n   startj " << startj << " j " << j << " end "
                          << d2 - 2 << std::endl;
#endif
                for (uint_t k = 0; k < d3; ++k) {
                    assert(offs_(i, j, k, d1, d2, d3) >= 0);
                    assert(offs_(i, j, k, d1, d2, d3) < d1 * d2 * d3);
                    out[offs_(i, j, k, d1, d2, d3)] =
                        4 * in[offs_(i, j, k, d1, d2, d3)] -
                        (in[offs_(i + 1, j, k, d1, d2, d3)] + in[offs_(i, j + 1, k, d1, d2, d3)] +
                            in[offs_(i - 1, j, k, d1, d2, d3)] + in[offs_(i, j - 1, k, d1, d2, d3)]);
                }
            }
        }
    }

    duration = (std::clock() - start) / (long double)CLOCKS_PER_SEC;

    print(out, d1, d2, d3, file_o);

    std::cout << "TIME " << duration << std::endl;

    return 0;
}

int main_block_inc(int argc, char **argv) {

    if (argc != 4) {
        std::cout << "Usage: basic_laplacian dimx dimy dimz\n where args are integer sizes of the data fields"
                  << std::endl;
        return 1;
    }

    /**
       The following steps are performed:

       - Definition of the domain:
    */
    uint_t d1 = atoi(argv[1]); /** d1 cells in the x direction (horizontal)*/
    uint_t d2 = atoi(argv[2]); /** d2 cells in the y direction (horizontal)*/
    uint_t d3 = atoi(argv[3]); /** d3 cells in the z direction (vertical)*/

    std::ofstream file_i("vanilla_block_inc_in");
    std::ofstream file_o("vanilla_block_inc_out");

    double *in = new double[d1 * d2 * d3];
    double *out = new double[d1 * d2 * d3];

    srand(12345);
    // DO NOT FUSE THESE LOOPS!!!
    for (uint_t i = 0; i < d1 * d2 * d3; ++i) {
        in[i] = -1.0 * rand();
    }

    srand(12345);
    for (uint_t i = 0; i < d1 * d2 * d3; ++i) {
        out[i] = -7.3 * rand();
    }

    print(out, d1, d2, d3, file_i);

    std::clock_t start;
    long double duration;
    start = std::clock();

    uint_t BI = 4;
    uint_t BJ = 4;

    uint_t NBI = (d1 - 4) / BI;
    uint_t NBJ = (d2 - 4) / BJ;
    {
        for (uint_t bi = 0; bi < NBI; ++bi) {
            for (uint_t bj = 0; bj < NBJ; ++bj) {
                uint_t starti = bi * BI + 2;
                uint_t startj = bj * BJ + 2;
                for (uint_t i = starti; i < starti + BI; ++i) {
                    for (uint_t j = startj; j < startj + BJ; ++j) {
#ifndef NDEBUG
                        std::cout << "B1"
                                  << " "
                                  << "starti " << starti << " "
                                  << " i " << i << " end " << starti + BI << "\n   startj " << startj << " j " << j
                                  << " end " << startj + BJ << std::endl;
#endif
                        double *po = out + offs_(i, j, 0, d1, d2, d3);
                        double *pi0 = in + offs_(i, j, 0, d1, d2, d3);
                        double *pi1 = in + offs_(i + 1, j, 0, d1, d2, d3);
                        double *pi2 = in + offs_(i, j + 1, 0, d1, d2, d3);
                        double *pi3 = in + offs_(i - 1, j, 0, d1, d2, d3);
                        double *pi4 = in + offs_(i, j - 1, 0, d1, d2, d3);
                        for (uint_t k = 0; k < d3; ++k) {
                            assert(offs_(i, j, k, d1, d2, d3) >= 0);
                            assert(offs_(i, j, k, d1, d2, d3) < d1 * d2 * d3);
                            *po = 4 * *pi0 - (*pi1 + *pi2 + *pi3 + *pi4);
                            ++po;
                            ++pi0;
                            ++pi1;
                            ++pi2;
                            ++pi3;
                            ++pi4;
                        }
                    }
                }
            }
        }

        for (uint_t bj = 0; bj < NBJ; ++bj) {
            uint_t starti = NBI * BI + 2;
            uint_t startj = bj * BJ + 2;
            for (uint_t i = starti; i < d1 - 2; ++i) {
                for (uint_t j = startj; j < startj + BJ; ++j) {
#ifndef NDEBUG
                    std::cout << "B2"
                              << " "
                              << "starti " << starti << " "
                              << " i " << i << " end " << d1 - 2 << "\n   startj " << startj << " j " << j << " end "
                              << startj + BJ << std::endl;
#endif
                    double *po = out + offs_(i, j, 0, d1, d2, d3);
                    double *pi0 = in + offs_(i, j, 0, d1, d2, d3);
                    double *pi1 = in + offs_(i + 1, j, 0, d1, d2, d3);
                    double *pi2 = in + offs_(i, j + 1, 0, d1, d2, d3);
                    double *pi3 = in + offs_(i - 1, j, 0, d1, d2, d3);
                    double *pi4 = in + offs_(i, j - 1, 0, d1, d2, d3);
                    for (uint_t k = 0; k < d3; ++k) {
                        assert(offs_(i, j, k, d1, d2, d3) >= 0);
                        assert(offs_(i, j, k, d1, d2, d3) < d1 * d2 * d3);
                        *po = 4 * *pi0 - (*pi1 + *pi2 + *pi3 + *pi4);
                        ++po;
                        ++pi0;
                        ++pi1;
                        ++pi2;
                        ++pi3;
                        ++pi4;
                    }
                }
            }
        }

        for (uint_t bi = 0; bi < NBI; ++bi) {
            uint_t starti = bi * BI + 2;
            uint_t startj = NBJ * BJ + 2;
            for (uint_t i = starti; i < starti + BI; ++i) {
                for (uint_t j = startj; j < d2 - 2; ++j) {
#ifndef NDEBUG
                    std::cout << "B3"
                              << " "
                              << "starti " << starti << " "
                              << " i " << i << " end " << starti + BI << "\n   startj " << startj << " j " << j
                              << " end " << d2 - 2 << std::endl;
#endif
                    double *po = out + offs_(i, j, 0, d1, d2, d3);
                    double *pi0 = in + offs_(i, j, 0, d1, d2, d3);
                    double *pi1 = in + offs_(i + 1, j, 0, d1, d2, d3);
                    double *pi2 = in + offs_(i, j + 1, 0, d1, d2, d3);
                    double *pi3 = in + offs_(i - 1, j, 0, d1, d2, d3);
                    double *pi4 = in + offs_(i, j - 1, 0, d1, d2, d3);
                    for (uint_t k = 0; k < d3; ++k) {
                        assert(offs_(i, j, k, d1, d2, d3) >= 0);
                        assert(offs_(i, j, k, d1, d2, d3) < d1 * d2 * d3);
                        *po = 4 * *pi0 - (*pi1 + *pi2 + *pi3 + *pi4);
                        ++po;
                        ++pi0;
                        ++pi1;
                        ++pi2;
                        ++pi3;
                        ++pi4;
                    }
                }
            }
        }

        uint_t starti = NBI * BI + 2;
        uint_t startj = NBJ * BJ + 2;
        for (uint_t i = starti; i < d1 - 2; ++i) {
            for (uint_t j = startj; j < d2 - 2; ++j) {
#ifndef NDEBUG
                std::cout << "B4"
                          << " "
                          << "starti " << starti << " "
                          << " i " << i << " end " << d1 - 2 << "\n   startj " << startj << " j " << j << " end "
                          << d2 - 2 << std::endl;
#endif
                double *po = out + offs_(i, j, 0, d1, d2, d3);
                double *pi0 = in + offs_(i, j, 0, d1, d2, d3);
                double *pi1 = in + offs_(i + 1, j, 0, d1, d2, d3);
                double *pi2 = in + offs_(i, j + 1, 0, d1, d2, d3);
                double *pi3 = in + offs_(i - 1, j, 0, d1, d2, d3);
                double *pi4 = in + offs_(i, j - 1, 0, d1, d2, d3);
                for (uint_t k = 0; k < d3; ++k) {
                    assert(offs_(i, j, k, d1, d2, d3) >= 0);
                    assert(offs_(i, j, k, d1, d2, d3) < d1 * d2 * d3);
                    *po = 4 * *pi0 - (*pi1 + *pi2 + *pi3 + *pi4);
                    ++po;
                    ++pi0;
                    ++pi1;
                    ++pi2;
                    ++pi3;
                    ++pi4;
                }
            }
        }
    }

    duration = (std::clock() - start) / (long double)CLOCKS_PER_SEC;

    print(out, d1, d2, d3, file_o);

    std::cout << "TIME " << duration << std::endl;

    return 0;
}

int main(int argc, char **argv) {

    std::cout << "******** NAIVE ********" << std::endl;
    main_naive(argc, argv);
    std::cout << "******** NAIVE_INC********" << std::endl;
    main_naive_inc(argc, argv);
    std::cout << "******** BLOCK ********" << std::endl;
    main_block(argc, argv);
    std::cout << "******** BLOCK_INC ********" << std::endl;
    main_block_inc(argc, argv);

    return 0;
}
