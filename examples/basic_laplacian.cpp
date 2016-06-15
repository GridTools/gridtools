/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include <gridtools.hpp>
#include <stencil_composition/stencil_composition.hpp>

int main_naive(int argc, char** argv) {

    if (argc != 4) {
        std::cout << "Usage: basic_laplacian dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    /**
       The following steps are performed:

       - Definition of the domain:
    */
    uint_t d1 = atoi(argv[1]); /** d1 cells in the x direction (horizontal)*/
    uint_t d2 = atoi(argv[2]); /** d2 cells in the y direction (horizontal)*/
    uint_t d3 = atoi(argv[3]); /** d3 cells in the z direction (vertical)*/

    typedef gridtools::backend<gridtools::enumtype::Host,gridtools::enumtype::Naive>::storage_type<double, gridtools::layout_map<0,1,2> >::type storage_type;

    std::ofstream file_i("basic_naive_in");
    std::ofstream file_o("basic_naive_out");

    storage_type in(d1,d2,d3,-1., "in");
    storage_type out(d1,d2,d3,-7.3, "out");
    out.print(file_i);

    boost::timer::cpu_timer time;
    for (int i=2; i < d1-2; ++i) {
        for (int j=2; j < d2-2; ++j) {
            for (int k=0; k < d3; ++k) {
                //std::cout << in(i,j,k) << std::endl;
                out(i,j,k) = 4 * in(i,j,k) -
                    (in( i+1, j, k) + in( i, j+1, k) +
                     in( i-1, j, k) + in( i, j-1, k));
            }
        }
    }
    boost::timer::cpu_times lapse_time = time.elapsed();

    out.print(file_o);

    std::cout << "TIME " << boost::timer::format(lapse_time) << std::endl;

    return 0;
}

int main_block(int argc, char** argv) {

    if (argc != 4) {
        std::cout << "Usage: basic_laplacian dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    /**
       The following steps are performed:

       - Definition of the domain:
    */
    uint_t d1 = atoi(argv[1]); /** d1 cells in the x direction (horizontal)*/
    uint_t d2 = atoi(argv[2]); /** d2 cells in the y direction (horizontal)*/
    uint_t d3 = atoi(argv[3]); /** d3 cells in the z direction (vertical)*/

    typedef gridtools::backend<gridtools::enumtype::Host,gridtools::enumtype::Naive>::storage_type<double, gridtools::layout_map<0,1,2> >::type storage_type;

    std::ofstream file_i("basic_block_in");
    std::ofstream file_o("basic_block_out");

    storage_type in(d1,d2,d3,-1., "in");
    storage_type out(d1,d2,d3,-7.3, "out");
    out.print(file_i);

    boost::timer::cpu_timer time;
    int BI = 4;
    int BJ = 4;

    int NBI = (d1-4)/BI;
    int NBJ = (d2-4)/BJ;
    {
        for (int bi = 0; bi < NBI; ++bi) {
            for (int bj = 0; bj < NBJ; ++bj) {
                int starti = bi*BI+2;
                int startj = bj*BJ+2;
                for (int i = starti; i < starti+BI; ++i) {
                    for (int j = startj; j < startj+BJ; ++j) {
#ifndef NDEBUG
                        std::cout << "B1" << " "
                                  << "starti " << starti << " "
                                  << " i " << i
                                  << " end " << starti+BI
                                  << "\n   startj " << startj
                                  << " j " << j
                                  << " end " << startj+BJ
                                  << std::endl;
#endif
                        for (int k = 0; k < d3; ++k) {
                            out(i,j,k) = 4 * in(i,j,k) -
                                (in( i+1, j, k) + in( i, j+1, k) +
                                 in( i-1, j, k) + in( i, j-1, k));
                        }
                    }
                }
            }
        }

        for (int bj = 0; bj < NBJ; ++bj) {
            int starti = NBI*BI+2;
            int startj = bj*BJ+2;
            for (int i = starti; i < d1-2; ++i) {
                for (int j = startj; j < startj+BJ; ++j) {
#ifndef NDEBUG
                    std::cout << "B2" << " "
                              << "starti " << starti << " "
                              << " i " << i
                              << " end " << d1-2
                              << "\n   startj " << startj
                              << " j " << j
                              << " end " << startj+BJ
                              << std::endl;
#endif
                    for (int k = 0; k < d3; ++k) {
                        out(i,j,k) = 4 * in(i,j,k) -
                            (in( i+1, j, k) + in( i, j+1, k) +
                             in( i-1, j, k) + in( i, j-1, k));
                    }
                }
            }
        }

        for (int bi = 0; bi < NBI; ++bi) {
            int starti = bi*BI+2;
            int startj = NBJ*BJ+2;
            for (int i = starti; i < starti+BI; ++i) {
                for (int j = startj; j < d2-2; ++j) {
#ifndef NDEBUG
                    std::cout << "B3" << " "
                              << "starti " << starti << " "
                              << " i " << i
                              << " end " << starti+BI
                              << "\n   startj " << startj
                              << " j " << j
                              << " end " << d2-2
                              << std::endl;
#endif
                    for (int k = 0; k < d3; ++k) {
                        out(i,j,k) = 4 * in(i,j,k) -
                            (in( i+1, j, k) + in( i, j+1, k) +
                             in( i-1, j, k) + in( i, j-1, k));
                    }
                }
            }
        }

        int starti = NBI*BI+2;
        int startj = NBJ*BJ+2;
        for (int i = starti; i < d1-2; ++i) {
            for (int j = startj; j < d2-2; ++j) {
#ifndef NDEBUG
                std::cout << "B4" << " "
                          << "starti " << starti << " "
                          << " i " << i
                          << " end " << d1-2
                          << "\n   startj " << startj
                          << " j " << j
                          << " end " << d2-2
                          << std::endl;
#endif
                for (int k = 0; k < d3; ++k) {
                    out(i,j,k) = 4 * in(i,j,k) -
                        (in( i+1, j, k) + in( i, j+1, k) +
                         in( i-1, j, k) + in( i, j-1, k));
                }
            }
        }
    }


    boost::timer::cpu_times lapse_time = time.elapsed();

    out.print(file_o);

    std::cout << "TIME " << boost::timer::format(lapse_time) << std::endl;

    return 0;
}

int main(int argc, char** argv) {

    std::cout << "******** NAIVE ********" << std::endl;
    main_naive(argc, argv);
    std::cout << "******** BLOCK ********" << std::endl;
    main_block(argc, argv);

    return 0;
}
