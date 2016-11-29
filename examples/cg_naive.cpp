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

    int dimx = atoi(argv[1]);
    int dimy = atoi(argv[2]);
    int dimz = atoi(argv[3]);
    int maxit = atoi(argv[4]);
    double eps = std::stod(argv[5]);
    int nrhs = atoi(argv[6]);

    //create timing class
    Timers timers;

    //TODO
    // Move definitions of storage type and partitioner here

    // q, r = 0
    //storage_type q    (metadata_, 0., "Vector for diagonal estimate");
    //storage_type r    (metadata_, 0., "Vector for diagonal estimate");
    
    // b = rnd
    //storage_type b    (metadata_, 0., "RHS for solver");
    
    //run GC solver for nrhs times
    for (int i = 0; i < nrhs; i++)
    {
        // x = inv(A) b
        cg_naive::solver(dimx, dimy, dimz, maxit, eps, &timers);
        
        // q = q + b .* x
    
        // r = r + b .* b
    }

    // d = q ./ r

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
