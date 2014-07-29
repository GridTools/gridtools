#include <iostream>
#include <fstream>
#include <boost/timer/timer.hpp>

#define offs_(i,j,k,n,m,l) ((i)*(m)*(l)+(j)*(l)+(k))

template <typename Stream>
void print(double* that, int n, int m, int l, Stream & stream) {
    //std::cout << "Printing " << name << std::endl;
    stream << "(" << n << "x"
           << m << "x"
           << l << ")"
           << std::endl;
    stream << "| j" << std::endl;
    stream << "| j" << std::endl;
    stream << "v j" << std::endl;
    stream << "---> k" << std::endl;

    int MI=12;
    int MJ=12;
    int MK=12;

    for (int i = 0; i < n; i += std::max(1,n/MI)) {
        for (int j = 0; j < m; j += std::max(1,m/MJ)) {
            for (int k = 0; k < l; k += std::max(1,l/MK)) {
                stream << "["/*("
                               << i << ","
                               << j << ","
                               << k << ")"*/
                       << that[offs_(i,j,k,n,m,l)] << "] ";
            }
            stream << std::endl;
        }
        stream << std::endl;
    }
    stream << std::endl;
}

int main_naive(int argc, char** argv) {

    if (argc != 4) {
        std::cout << "Usage: basic_laplacian dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    /**
	The following steps are performed:

	- Definition of the domain:
    */
    int d1 = atoi(argv[1]); /** d1 cells in the x direction (horizontal)*/
    int d2 = atoi(argv[2]); /** d2 cells in the y direction (horizontal)*/
    int d3 = atoi(argv[3]); /** d3 cells in the z direction (vertical)*/

    std::ofstream file_i("vanilla_naive_in");
    std::ofstream file_o("vanilla_naive_out");

    double* in = new double[d1*d2*d3];
    double* out = new double[d1*d2*d3];

    srand(12345);
    // DO NOT FUSE THESE LOOPS!!!
    for (int i = 0; i < d1*d2*d3; ++i) {
        in[i] = -1.0 * rand();
    }

    srand(12345);
    for (int i = 0; i < d1*d2*d3; ++i) {
        out[i] = -7.3 * rand();
    }

    print(out, d1, d2, d3, file_i);

    boost::timer::cpu_timer time;
    for (int i=2; i < d1-2; ++i) {
        for (int j=2; j < d2-2; ++j) {
            for (int k=0; k < d3; ++k) {
                //std::cout << in(i,j,k) << std::endl;
                assert(offs_(i,j,k,d1,d2,d3) >= 0);
                assert(offs_(i,j,k,d1,d2,d3) < d1*d2*d3);
                // std::cout << i << ", " << j << ", " << k << " - " << d1 << ", " << d2 << ", " << d3 << " -- " << offs_(i,j,k,d1,d2,d3) << " " << d1*d2*d3 << std::endl;
                // std::cout << i << ", " << j << ", " << k << " - " << d1 << ", " << d2 << ", " << d3 << " -- " << offs_(i+1,j,k,d1,d2,d3) << " " << d1*d2*d3 << std::endl;
                // std::cout << i << ", " << j << ", " << k << " - " << d1 << ", " << d2 << ", " << d3 << " -- " << offs_(i-1,j,k,d1,d2,d3) << " " << d1*d2*d3 << std::endl;
                out[offs_(i,j,k,d1,d2,d3)] = 
                    4 * in[offs_(i,j,k,d1,d2,d3)] -
                    (in[offs_(i+1,j,k,d1,d2,d3)] + in[offs_(i,j+1,k,d1,d2,d3)] +
                     in[offs_(i-1,j,k,d1,d2,d3)] + in[offs_(i,j-1,k,d1,d2,d3)]);
            }
        }
    }
    boost::timer::cpu_times lapse_time = time.elapsed();

    print(out, d1, d2, d3, file_o);

    std::cout << "TIME " << boost::timer::format(lapse_time) << std::endl;

    delete[] in;
    delete[] out;

    return 0;
}

int main_naive_inc(int argc, char** argv) {

    if (argc != 4) {
        std::cout << "Usage: basic_laplacian dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    /**
	The following steps are performed:

	- Definition of the domain:
    */
    int d1 = atoi(argv[1]); /** d1 cells in the x direction (horizontal)*/
    int d2 = atoi(argv[2]); /** d2 cells in the y direction (horizontal)*/
    int d3 = atoi(argv[3]); /** d3 cells in the z direction (vertical)*/

    std::ofstream file_i("vanilla_naive_inc_in");
    std::ofstream file_o("vanilla_naive_inc_out");

    double* in = new double[d1*d2*d3];
    double* out = new double[d1*d2*d3];

    srand(12345);
    // DO NOT FUSE THESE LOOPS!!!
    for (int i = 0; i < d1*d2*d3; ++i) {
        in[i] = -1.0 * rand();
    }

    srand(12345);
    for (int i = 0; i < d1*d2*d3; ++i) {
        out[i] = -7.3 * rand();
    }

    print(out, d1, d2, d3, file_i);

    boost::timer::cpu_timer time;
    for (int i=2; i < d1-2; ++i) {
        for (int j=2; j < d2-2; ++j) {
            double* po = out + offs_(i,j,0,d1,d2,d3);
            double* pi0 = in + offs_(i,j,0,d1,d2,d3);
            double* pi1 = in + offs_(i+1,j,0,d1,d2,d3);
            double* pi2 = in + offs_(i,j+1,0,d1,d2,d3);
            double* pi3 = in + offs_(i-1,j,0,d1,d2,d3);
            double* pi4 = in + offs_(i,j-1,0,d1,d2,d3);
            for (int k=0; k < d3; ++k) {
                //std::cout << in(i,j,k) << std::endl;
                assert(offs_(i,j,k,d1,d2,d3) >= 0);
                assert(offs_(i,j,k,d1,d2,d3) < d1*d2*d3);
                // std::cout << i << ", " << j << ", " << k << " - " << d1 << ", " << d2 << ", " << d3 << " -- " << offs_(i,j,k,d1,d2,d3) << " " << d1*d2*d3 << std::endl;
                // std::cout << i << ", " << j << ", " << k << " - " << d1 << ", " << d2 << ", " << d3 << " -- " << offs_(i+1,j,k,d1,d2,d3) << " " << d1*d2*d3 << std::endl;
                // std::cout << i << ", " << j << ", " << k << " - " << d1 << ", " << d2 << ", " << d3 << " -- " << offs_(i-1,j,k,d1,d2,d3) << " " << d1*d2*d3 << std::endl;
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
    boost::timer::cpu_times lapse_time = time.elapsed();

    print(out, d1, d2, d3, file_o);

    std::cout << "TIME " << boost::timer::format(lapse_time) << std::endl;

    delete[] in;
    delete[] out;

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
    int d1 = atoi(argv[1]); /** d1 cells in the x direction (horizontal)*/
    int d2 = atoi(argv[2]); /** d2 cells in the y direction (horizontal)*/
    int d3 = atoi(argv[3]); /** d3 cells in the z direction (vertical)*/

    std::ofstream file_i("vanilla_block_in");
    std::ofstream file_o("vanilla_block_out");

    double* in = new double[d1*d2*d3];
    double* out = new double[d1*d2*d3];

    srand(12345);
    // DO NOT FUSE THESE LOOPS!!!
    for (int i = 0; i < d1*d2*d3; ++i) {
        in[i] = -1.0 * rand();
    }

    srand(12345);
    for (int i = 0; i < d1*d2*d3; ++i) {
        out[i] = -7.3 * rand();
    }

    print(out, d1, d2, d3, file_i);

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
                            assert(offs_(i,j,k,d1,d2,d3) >= 0);
                            assert(offs_(i,j,k,d1,d2,d3) < d1*d2*d3);
                            out[offs_(i,j,k,d1,d2,d3)] = 
                                4 * in[offs_(i,j,k,d1,d2,d3)] -
                                (in[offs_(i+1,j,k,d1,d2,d3)] + in[offs_(i,j+1,k,d1,d2,d3)] +
                                 in[offs_(i-1,j,k,d1,d2,d3)] + in[offs_(i,j-1,k,d1,d2,d3)]);
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
                        assert(offs_(i,j,k,d1,d2,d3) >= 0);
                        assert(offs_(i,j,k,d1,d2,d3) < d1*d2*d3);
                        out[offs_(i,j,k,d1,d2,d3)] = 
                            4 * in[offs_(i,j,k,d1,d2,d3)] -
                            (in[offs_(i+1,j,k,d1,d2,d3)] + in[offs_(i,j+1,k,d1,d2,d3)] +
                             in[offs_(i-1,j,k,d1,d2,d3)] + in[offs_(i,j-1,k,d1,d2,d3)]);
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
                        assert(offs_(i,j,k,d1,d2,d3) >= 0);
                        assert(offs_(i,j,k,d1,d2,d3) < d1*d2*d3);
                        out[offs_(i,j,k,d1,d2,d3)] = 
                            4 * in[offs_(i,j,k,d1,d2,d3)] -
                            (in[offs_(i+1,j,k,d1,d2,d3)] + in[offs_(i,j+1,k,d1,d2,d3)] +
                             in[offs_(i-1,j,k,d1,d2,d3)] + in[offs_(i,j-1,k,d1,d2,d3)]);
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
                    assert(offs_(i,j,k,d1,d2,d3) >= 0);
                    assert(offs_(i,j,k,d1,d2,d3) < d1*d2*d3);
                    out[offs_(i,j,k,d1,d2,d3)] = 
                        4 * in[offs_(i,j,k,d1,d2,d3)] -
                        (in[offs_(i+1,j,k,d1,d2,d3)] + in[offs_(i,j+1,k,d1,d2,d3)] +
                         in[offs_(i-1,j,k,d1,d2,d3)] + in[offs_(i,j-1,k,d1,d2,d3)]);
                }
            }
        }
    }
    

    boost::timer::cpu_times lapse_time = time.elapsed();

    print(out, d1, d2, d3, file_o);

    std::cout << "TIME " << boost::timer::format(lapse_time) << std::endl;

    return 0;
}

int main_block_inc(int argc, char** argv) {

    if (argc != 4) {
        std::cout << "Usage: basic_laplacian dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    /**
	The following steps are performed:

	- Definition of the domain:
    */
    int d1 = atoi(argv[1]); /** d1 cells in the x direction (horizontal)*/
    int d2 = atoi(argv[2]); /** d2 cells in the y direction (horizontal)*/
    int d3 = atoi(argv[3]); /** d3 cells in the z direction (vertical)*/

    std::ofstream file_i("vanilla_block_inc_in");
    std::ofstream file_o("vanilla_block_inc_out");

    double* in = new double[d1*d2*d3];
    double* out = new double[d1*d2*d3];


    srand(12345);
    // DO NOT FUSE THESE LOOPS!!!
    for (int i = 0; i < d1*d2*d3; ++i) {
        in[i] = -1.0 * rand();
    }

    srand(12345);
    for (int i = 0; i < d1*d2*d3; ++i) {
        out[i] = -7.3 * rand();
    }

    print(out, d1, d2, d3, file_i);

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
                        double* po = out + offs_(i,j,0,d1,d2,d3);
                        double* pi0 = in + offs_(i,j,0,d1,d2,d3);
                        double* pi1 = in + offs_(i+1,j,0,d1,d2,d3);
                        double* pi2 = in + offs_(i,j+1,0,d1,d2,d3);
                        double* pi3 = in + offs_(i-1,j,0,d1,d2,d3);
                        double* pi4 = in + offs_(i,j-1,0,d1,d2,d3);
                        for (int k = 0; k < d3; ++k) {
                            assert(offs_(i,j,k,d1,d2,d3) >= 0);
                            assert(offs_(i,j,k,d1,d2,d3) < d1*d2*d3);
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
                    double* po = out + offs_(i,j,0,d1,d2,d3);
                    double* pi0 = in + offs_(i,j,0,d1,d2,d3);
                    double* pi1 = in + offs_(i+1,j,0,d1,d2,d3);
                    double* pi2 = in + offs_(i,j+1,0,d1,d2,d3);
                    double* pi3 = in + offs_(i-1,j,0,d1,d2,d3);
                    double* pi4 = in + offs_(i,j-1,0,d1,d2,d3);
                    for (int k = 0; k < d3; ++k) {
                        assert(offs_(i,j,k,d1,d2,d3) >= 0);
                        assert(offs_(i,j,k,d1,d2,d3) < d1*d2*d3);
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
                    double* po = out + offs_(i,j,0,d1,d2,d3);
                    double* pi0 = in + offs_(i,j,0,d1,d2,d3);
                    double* pi1 = in + offs_(i+1,j,0,d1,d2,d3);
                    double* pi2 = in + offs_(i,j+1,0,d1,d2,d3);
                    double* pi3 = in + offs_(i-1,j,0,d1,d2,d3);
                    double* pi4 = in + offs_(i,j-1,0,d1,d2,d3);
                    for (int k = 0; k < d3; ++k) {
                        assert(offs_(i,j,k,d1,d2,d3) >= 0);
                        assert(offs_(i,j,k,d1,d2,d3) < d1*d2*d3);
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
                double* po = out + offs_(i,j,0,d1,d2,d3);
                double* pi0 = in + offs_(i,j,0,d1,d2,d3);
                double* pi1 = in + offs_(i+1,j,0,d1,d2,d3);
                double* pi2 = in + offs_(i,j+1,0,d1,d2,d3);
                double* pi3 = in + offs_(i-1,j,0,d1,d2,d3);
                double* pi4 = in + offs_(i,j-1,0,d1,d2,d3);
                for (int k = 0; k < d3; ++k) {
                    assert(offs_(i,j,k,d1,d2,d3) >= 0);
                    assert(offs_(i,j,k,d1,d2,d3) < d1*d2*d3);
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
    

    boost::timer::cpu_times lapse_time = time.elapsed();

    print(out, d1, d2, d3, file_o);

    std::cout << "TIME " << boost::timer::format(lapse_time) << std::endl;

    return 0;
}

int main(int argc, char** argv) {

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
