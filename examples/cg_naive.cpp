#include "cg_naive.hpp"

int main(int argc, char** argv)
{
#ifdef CXX11_ENABLED
    // Initialize MPI
    gridtools::GCL_Init();

    if (argc != 7) {
        if (gridtools::PID == 0)
        std::cout << "Usage: cg_naive<whatever> dimx dimy dimz maxit eps nrhs,\nwhere args are integer sizes of the data fields, max number of iterations, eps is required tolerance and nsamples is number of RHS" << std::endl;
        return 1;
    }

    //create timing class
    Timers timers;

    //run GC solver for NS samples
    int NS = atoi(argv[6]);
    for (int i = 0; i < NS; i++)
    {
        cg_naive::solver(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), std::stod(argv[5]), &timers);
    }

    //print timing info
    if (gridtools::PID == 0)
    {
        timers.print_timers();
    }
    
    gridtools::GCL_Finalize();
#else
    assert(false);
    return -1;
#endif
}
