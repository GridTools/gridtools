#include "cg_naive.hpp"

using namespace cg_naive;

// I am using another add functor because if the add_functor
// from .hpp file is used it produces a compilation error,
// which is probably a bug in gcc compiler
// https://stackoverflow.com/questions/10671406/c-confusing-attribute-name-for-member-template
struct add_functor_simple{
    typedef accessor<0, enumtype::inout, extent<0,0,0,0> > c;
    typedef accessor<1, enumtype::in, extent<0,0,0,0> > a;
    typedef accessor<2, enumtype::in, extent<0,0,0,0> > b;
    typedef boost::mpl::vector<c,a,b> arg_list;

    template <typename Domain>
        GT_FUNCTION
        static void Do(Domain const & dom, x_interval) {
            dom(c{}) = dom(a{}) + dom(b{});
        }
};

struct div_functor{
    typedef accessor<0, enumtype::inout, extent<0,0,0,0> > c;
    typedef accessor<1, enumtype::in, extent<0,0,0,0> > a;
    typedef accessor<2, enumtype::in, extent<0,0,0,0> > b;
    typedef boost::mpl::vector<c,a,b> arg_list;

    template <typename Domain>
        GT_FUNCTION
        static void Do(Domain const & dom, x_interval) {
            dom(c{}) = dom(a{}) / dom(b{});
        }
};

int main(int argc, char** argv)
{
#ifdef CXX11_ENABLED
    // Initialize MPI
    gridtools::GCL_Init();
    const int MASTER = 0;

    if (argc != 7) {
        if (gridtools::PID == MASTER) std::cout << "Usage: cg_naive<whatever> dimx dimy dimz maxit eps nrhs,\nwhere args are integer sizes of the data fields, max number of iterations, eps is required tolerance and nsamples is number of RHS" << std::endl;
        return 1;
    }

    int dimx = atoi(argv[1]);
    int dimy = atoi(argv[2]);
    int dimz = atoi(argv[3]);
    int maxit = atoi(argv[4]); //max number of iterations for solver
    double eps = std::stod(argv[5]); //solver tolerance
    int nrhs = atoi(argv[6]); //number of samples for stochastic estimate
    
    // create timing class
    Timers timers;

    // cg solver class
    CGsolver cg(dimx, dimy, dimz);

    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // processor grid dimensions
    int N1 = cg.dimensions[0];
    int N2 = cg.dimensions[1];

    //size of arrays to hold right-hand sides
    long int global_domain_size = dimx*dimy*dimz;
    long int count = global_domain_size*nrhs;
    long int local_count = count / mpi_size;
    
    auto metadata_ = cg.meta_->get_metadata();
    int dimx_local = metadata_.template dims<0>() - 2;
    int dimy_local = metadata_.template dims<1>() - 2;
    int dimz_local = metadata_.template dims<2>() - 2;
    long int local_domain_size = dimx_local * dimy_local * dimz_local;

    //assert
    if (local_count != (local_domain_size*nrhs))
    {
        printf("lds*nrhs= %ld %ld %ld\n",local_domain_size, nrhs, local_domain_size*nrhs);
        printf("local_count != local_domain_size*nrhs\n %ld %ld\n", local_count, local_domain_size*nrhs);
        return -1;
    }

    if (PID == MASTER) printf("Computing for domain size %dx%dx%d and %d random samples using %d processes (%dx%d).\nRequired tolerance for CG is %e with allowed %d max iterations.\n", dimx, dimy, dimz, nrhs, mpi_size, N1, N2, eps, maxit);

    double *samples;
    double *samples_local = new double[local_count];
    if (samples_local == NULL)
    {
        printf("Error in new samples_local[]\n");
        return -1;
    }

    // generate random RHS and scatter them to processes
    if (PID == MASTER)
    {
        samples = new double[local_count];
        if (samples == NULL)
        {
            printf("Error in new samples[]\n");
            return -1;
        }
        
        //std::srand(std::time(0));
        std::srand(131867);

        //generate RHS for MASTER process
        for (long int i = 0; i < local_count; i++)
        {
           samples_local[i] = (std::rand() / (double)RAND_MAX) > 0.5 ? 1.0 : -1.0;
        }

        //generage RHS for child processes
        for (int p=1; p < mpi_size; p++)
        {
            for (long int i = 0; i < local_count; i++)
            {
                samples[i] = (std::rand() / (double)RAND_MAX) > 0.5 ? 1.0 : -1.0;
            }
            MPI_Send(samples, local_count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Recv(samples_local, local_count, MPI_DOUBLE, MASTER, 0,  MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    if (PID == MASTER) delete [] samples;

    // local domains
    storage_type x    (metadata_, 0., "Solution vector");
    storage_type b    (metadata_, 0., "RHS vector");
    storage_type q    (metadata_, 0., "Helper vector for diagonal estimate");
    storage_type r    (metadata_, 0., "Helper vector for diagonal estimate");
    storage_type d    (metadata_, 0., "Vector for diagonal estimate");
    
    // prepare structures for GridTools kernels
    typedef arg<0, storage_type > p_q;
    typedef arg<1, storage_type > p_r;
    typedef arg<2, storage_type > p_b;
    typedef arg<3, storage_type > p_x;
    typedef arg<4, storage_type > p_tmp; //used as temporary grid for products

    typedef boost::mpl::vector<
            p_q,
            p_r,
            p_b,
            p_x,
            p_tmp> accessor_list_qr;
    
    typedef arg<0, storage_type > p_d;
    typedef arg<1, storage_type > p_q1;
    typedef arg<2, storage_type > p_r1;
    typedef boost::mpl::vector<
            p_d,
            p_q1,
            p_r1 > accessor_list_d;
    
    // Construction of the domain for step phase
    gridtools::domain_type<accessor_list_qr> domain_qr
        (boost::fusion::make_vector(&q, &r, &b, &x, &d)); //q = q + b.*x, r = r + b.*b, domain d used as tmp
    
    gridtools::domain_type<accessor_list_d> domain_d
        (boost::fusion::make_vector(&d, &q, &r)); //d = q ./ r

    // Instantiate stencil to perform initialization step of CG
    auto sedi = gridtools::make_computation<gridtools::BACKEND>
        (
         domain_qr, *(cg.coords3d7pt),
         gridtools::make_mss // mss_descriptor
         (
          execute<forward>(),
          gridtools::make_esf<product_functor>(p_tmp(), p_b(), p_x()), // tmp = b .* x
          gridtools::make_esf<add_functor_simple>(p_q(), p_q(), p_tmp()), // q = q + tmp
          gridtools::make_esf<product_functor>(p_tmp(), p_b(), p_b()), // tmp = b .* b
          gridtools::make_esf<add_functor_simple>(p_r(), p_r(), p_tmp()) // r = r + tmp
         )
        );

    auto inverse = gridtools::make_computation<gridtools::BACKEND>
        (
         domain_d, *(cg.coords3d7pt),
         gridtools::make_mss // mss_descriptor
         (
          execute<forward>(),
          gridtools::make_esf<div_functor>(p_d(), p_q1(), p_r1()) // d = q ./ r
         )
        );

    // Start timer
    timers.start(Timers::TIMER_GLOBAL);

    /**
     *  run GC solver for each sample
     *  =============================
     */
    for (int ii = 0; ii < nrhs; ii++)
    {
        double *rhs = &samples_local[ii * local_domain_size];

        // Initialize the local RHS vector domain (exclude halo layer)
        int idx = 0;
        for (uint_t i = 1; i < dimx_local + 1  ; ++i)
            for (uint_t j = 1; j < dimy_local + 1; ++j)
                for (uint_t k = 1; k < dimz_local + 1 ; ++k)
                {
                    x(i,j,k) = 0;
                    b(i,j,k) = rhs[idx++];
                }

        // x = inv(A) b
        bool converged = cg.solver(x, b, maxit, eps, timers);
        if (!converged)
        {
            if(PID == MASTER) printf("CG did not converge to the specified tolerance %e\n",eps);
            //return -1;
        }
        
        // q = q + b .* x
        // r = r + b .* b // this could be optimized by setting it to nrhs*ones(dim_global)
        sedi->ready();
        sedi->steady();
        sedi->run();
        sedi->finalize();
        
    }

    // d = q ./ r
    inverse->ready();
    inverse->steady();
    inverse->run();
    inverse->finalize();

    delete [] samples_local;
   
   // Stop timer
   timers.stop(Timers::TIMER_GLOBAL);

    /**
     *  Extract and print the diagonal estimate
     *  =======================================
     */
#if 0 
    double *estimator;
    if (PID == MASTER)
    {
        estimator = new double [global_domain_size];
        if (estimator == NULL)
        {
            printf("Error in new estimator[]\n");
            return -1;
        }

    }

    // remove the boundary layers from the local domain 
    double *estimator_local = new double[local_domain_size];
    if (estimator_local == NULL)
    {
        printf("Error in new estimator_local[]\n");
        return -1;
    }

    int idx = 0;
    for (uint_t k = 1; k < dimz_local + 1 ; ++k)
        for (uint_t j = 1; j < dimy_local + 1; ++j)
            for (uint_t i = 1; i < dimx_local + 1  ; ++i)
            {
                estimator_local[idx++] = d(i,j,k); 
            }

    // gather the whole estimator domain to master node
    // domains are put one after each other, not in natural ordering!!!
    MPI_Gather(estimator_local, local_domain_size, MPI_DOUBLE, estimator, local_domain_size, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    delete [] estimator_local;

    //print the diagonal estimate
    if (PID == MASTER)
    {
        char* outfilename = new char[500];
        sprintf(outfilename, "/home/kardoj/gridtools/diags/diagSEDI_%dx%dx%d_%5.1e_%d_%d.txt", dimx, dimy, dimz, eps, nrhs, mpi_size);
        printf("Saving diagonal estimate into file: %s\n",outfilename);
        FILE* fileout;
        fileout = fopen(outfilename, "w");
        

        // loop over processor grix px,py,pz
        // and over local domain x,y,z
        for (int z = 0; z < dimz; z++)
        {
          for (int px = 0; px < N1; px++) //Gridtools has inverted x,y axes of processor grid
          {
              for (int y = 0; y < dimy_local; y++)
              {
                for (int py = 0; py < N2; py++)
                {
                    int pidx = px + py * N1;

                    for (int x = 0; x < dimx_local; x++)
                    {
                        int offset = pidx*local_domain_size + z*dimx_local*dimy_local + y*dimx_local + x; // index in gathered domain
                        int idx = (py*dimx_local + x) + (px*dimy_local + y)*dimx + z*dimx*dimy; //row of the A
                        if(offset >= global_domain_size) {printf("OFFSET > domain size\n");}
                        printf ("Diagonal element (%d, %d): %32.24e\n", idx, idx, estimator[offset]);
                        fprintf (fileout, "%32.24e\n", estimator[offset]);
                    }
                }
              }
            }
          }
        fclose(fileout);
        delete [] estimator;
    }

#endif

    //print timing info
    if (gridtools::PID == MASTER)
    {
        timers.print_timers();
    }
    
    gridtools::GCL_Finalize();

#else
    assert(false);
    return -1;
#endif
}
