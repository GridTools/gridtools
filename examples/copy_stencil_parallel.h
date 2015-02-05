
#pragma once

#include <gridtools.h>
#ifdef CUDA_EXAMPLE
#include <stencil-composition/backend_cuda.h>
#else
#include <stencil-composition/backend_host.h>
#endif
#include <storage/partitioner_trivial.h>
#include <storage/parallel_storage.h>

#include <stencil-composition/interval.h>
#include <stencil-composition/make_computation.h>

#include <communication/halo_exchange.h>
/*
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend
*/

using gridtools::level;
using gridtools::arg_type;
using gridtools::range;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;


namespace copy_stencil{


/* This is the data type of the elements of the data arrays.
 */
struct triple_t {
  int x,y,z;
  triple_t(int a, int b, int c): x(a), y(b), z(c) {}
  triple_t(): x(-1), y(-1), z(-1) {}
};

std::ostream& operator<<(std::ostream &s, triple_t const & t) {
  return s << " ("
           << t.x << ", "
           << t.y << ", "
           << t.z << ") ";
}

bool operator==(triple_t const & a, triple_t const & b) {
  return (a.x == b.x &&
          a.y == b.y &&
          a.z == b.z);
}

bool operator!=(triple_t const & a, triple_t const & b) {
  return !(a==b);
}

/* Just and utility to print values
 */
void printbuff(std::ostream &file, triple_t* a, int d1, int d2, int d3) {
  if (d1<6 && d2<6 && d3<6) {
    file << "------------\n";
    for (int ii=0; ii<d1; ++ii) {
      file << "|";
      for (int jj=0; jj<d2; ++jj) {
        for (int kk=0; kk<d2; ++kk) {
          file << a[d1*d2*kk+ii*d2+jj];
        }
        file << "|\n";
      }
      file << "\n\n";
    }
    file << "------------\n\n";
  }
}

// This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

// These are the stencil operators that compose the multistage stencil in this test
    struct copy_functor {
#ifdef CXX11_ENABLED
	typedef arg_type<0, range<0,0,0,0>, 4> in;
	typedef boost::mpl::vector<in> arg_list;
	typedef Dimension<4> time;
#else
	typedef const arg_type<0>::type in;
	typedef arg_type<1>::type out;
	typedef boost::mpl::vector<in,out> arg_list;
#endif
	/* static const auto expression=in(1,0,0)-out(); */

	template <typename Evaluation>
	GT_FUNCTION
	static void Do(Evaluation const & eval, x_interval) {
#ifdef CXX11_ENABLED
	    eval(in(0,0,0,1))
#else
		eval(out())
#endif
		=eval(in());
	}
    };

/*
 * The following operators and structs are for debugging only
 */
    std::ostream& operator<<(std::ostream& s, copy_functor const) {
	return s << "copy_functor";
    }

    void handle_error(int_t)
    {std::cout<<"error"<<std::endl;}

    // uint_t low_bound(int pid, int ntasks, uint_t dimension)
    // {
    //     if (pid==0)
    //         return 0;
    //     div_t value=std::div(dimension,ntasks);
    //     return ((int)(value.quot/**(pid)*/) + (int)((value.rem>=pid) ? /*(pid)*/1 : 0/*value.rem*/));
    // }


    int high_bound(int pid, int ntasks, uint_t dimension)
    {
	// if (pid==ntasks-1)
	//     return dimension-1;
	div_t value=std::div(dimension,ntasks);
	std::cout<<"quotient: "<< value.quot <<std::endl;
	std::cout<<"pid*4: "<< value.quot <<std::endl;
	std::cout<<"offset: "<<(int)((value.rem>pid) ? (pid+1): value.rem) <<std::endl;
	std::cout<<"ret val: "<< ((int)(value.quot*(pid+1)) + (int)((value.rem>pid) ? (pid+1) : value.rem)) <<std::endl;
	return ((int)(value.quot/**(pid+1)*/) + (int)((value.rem>pid) ? (/*pid+*/1) : value.rem));
    }

    bool test(uint_t x, uint_t y, uint_t z) {

#ifdef CUDA_EXAMPLE
#define BACKEND backend<Cuda, Naive >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, Block >
#else
#define BACKEND backend<Host, Naive >
#endif
#endif
	//                   strides  1 x xy
	//                      dims  x y z
	typedef gridtools::layout_map<0,1,2> layout_t;
	typedef gridtools::BACKEND::storage_type<float_type, layout_t >::type storage_type;
	//typedef storage_type::basic_type integrator_type;
	/* typedef extend<storage_type::basic_type, 2> integrator_type; */
#ifdef CXX11_ENABLED							\
    /* The nice interface does not compile today (CUDA 6.5) with nvcc (C++11 support not complete yet)*/
#ifdef __CUDACC__
//pointless and tedious syntax, temporary while thinking/waiting for an alternative like below
	typedef base_storage<Cuda, float_type, layout_t, false ,2> base_type1;
	typedef extend_width<base_type1, 0>  extended_type;
	typedef storage<extend_dim<extended_type, extended_type> >  integrator_type;
#else
	typedef field<storage_type, 2>::type  integrator_type;
#endif
#endif

	uint_t d1 = x;
	uint_t d2 = y;
	uint_t d3 = z;
	uint_t H  = 1;

	int pid;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	int nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	std::cout << pid << " " << nprocs << "\n";

	std::stringstream ss;
	ss << pid;

	std::string filename = "out" + ss.str() + ".txt";

	std::cout << filename << std::endl;
	std::ofstream file(filename.c_str());

	file << pid << "  " << nprocs << "\n";

	MPI_Comm CartComm;
	int dims[3] = { 0, 0, 0 };
	MPI_Dims_create(nprocs, 3, dims);
	int period[3] = { 1, 1, 1 };

	file << "@" << pid << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1]
	     << " - " << dims[2] << "\n";

	MPI_Cart_create(MPI_COMM_WORLD, 3, dims, period, false, &CartComm);
	int coordinates[3] = { 0, 0, 0 };
	MPI_Cart_get(CartComm, 3, dims, period, coordinates);

	typedef gridtools::halo_exchange_dynamic_ut<gridtools::layout_map<0, 1, 2>,
						    gridtools::layout_map<0, 1, 2>, integrator_type::original_storage::pointer_type::pointee_t, 3, gridtools::gcl_cpu, gridtools::version_manual> pattern_type;

	pattern_type he(pattern_type::grid_type::period_type(true, true, true), CartComm);

	int low_bound_i=0;//low_bound(coordinates[0], dims[0], d1-H-H);
	int low_bound_j=0;//low_bound(coordinates[1], dims[1], d2-H-H);
	int high_bound_i=high_bound(coordinates[0], dims[0], d1-H-H);
	int high_bound_j=high_bound(coordinates[1], dims[1], d2-H-H);

        std::cout<<"d1= "<<d1<<"d2= "<<d2<<std::endl;
	std::cout<< " "<<H<< " "<<H<< " "<<H<< " "<<high_bound_i+H-1<< " "<<high_bound_i-low_bound_i+H+H<< " "<<std::endl;
	std::cout<< " "<<H<< " "<<H<< " "<<H<< " "<<high_bound_j+H-1<< " "<<high_bound_j-low_bound_j+H+H<< " "<<std::endl;
	he.add_halo<0>(H, H, H, high_bound_i+H-1, high_bound_i+H+H);
	he.add_halo<1>(H, H, H, high_bound_j+H-1, high_bound_j+H+H);
	he.add_halo<2>(0, 0, 0, d3 - 1, d3);

        he.setup(2);

	//out.print();

	// Definition of placeholders. The order of them reflect the order the user will deal with them
	// especially the non-temporary ones, in the construction of the domain
#ifdef CXX11_ENABLED
	typedef arg<0, integrator_type > p_in;
	typedef boost::mpl::vector<p_in> arg_type_list;
#else
	typedef arg<0, storage_type> p_in;
	typedef arg<1, storage_type> p_out;
	// An array of placeholders to be passed to the domain
	// I'm using mpl::vector, but the final API should look slightly simpler
	typedef boost::mpl::vector<p_in, p_out> arg_type_list;
#endif
	/* typedef arg<1, integrator_type > p_out; */

	// Definition of the actual data fields that are used for input/output
#ifdef CXX11_ENABLED
        ushort_t halo[3]={H,H,H};
        partitioner_trivial<integrator_type> part(coordinates, dims, halo);
	parallel_storage<integrator_type> in(part, d1, d2, d3);
	//integrator_type in(d1,d2,d3);
	integrator_type::original_storage::pointer_type  init1(d1*d2*d3);
	integrator_type::original_storage::pointer_type  init2(d1*d2*d3);
	in.push_front<0>(init1, 1.5);
	in.push_front<0>(init2, -1.5);
#else
	storage_type in(d1,d2,d3,-3.5,"in");
	storage_type out(d1,d2,d3,1.5,"out");
#endif

	for(uint_t i=0; i<in.template dims<0>(); ++i)
	    for(uint_t j=0; j<in.template dims<1>(); ++j)
		for(uint_t k=0; k<in.template dims<2>(); ++k)
		{
		    in(i, j, k)=i+j+k+coordinates[0]*100+coordinates[1]*200+coordinates[2]*300;
		}


	// construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
	// It must be noted that the only fields to be passed to the constructor are the non-temporary.
	// The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)
#ifdef CXX11_ENABLED
	gridtools::domain_type<arg_type_list> domain
	    (boost::fusion::make_vector(&in));
#else
	gridtools::domain_type<arg_type_list> domain
	    (boost::fusion::make_vector(&in, &out));
#endif
	// Definition of the physical dimensions of the problem.
	// The constructor takes the horizontal plane dimensions,
	// while the vertical ones are set according the the axis property soon after
	// gridtools::coordinates<axis> coords(2,d1-2,2,d2-2);
	// uint_t di[5] = {0, 0, 0, d1, d1};
	// uint_t dj[5] = {0, 0, 0, d2, d2};
	uint_t di[5] = {H, H, H, high_bound_i+H-1, high_bound_i+H+H};
	uint_t dj[5] = {H, H, H, high_bound_j+H-1, high_bound_j+H+H};

	// std::cout<< " "<<H<< " "<<H<< " "<<H<< " "<<high_bound_i+H-1<< " "<<high_bound_i+H+H<< " "<<std::endl;
	// std::cout<< " "<<H<< " "<<H<< " "<<H<< " "<<high_bound_j+H-1<< " "<<high_bound_j+H+H<< " "<<std::endl;

	gridtools::coordinates<axis> coords(di, dj);
	coords.value_list[0] = 0;
	coords.value_list[1] = d3-1;

	/*
	  Here we do lot of stuff
	  1) We pass to the intermediate representation ::run function the description
	  of the stencil, which is a multi-stage stencil (mss)
	  The mss includes (in order of execution) a laplacian, two fluxes which are independent
	  and a final step that is the out_function
	  2) The logical physical domain with the fields to use
	  3) The actual domain dimensions
	*/

// \todo simplify the following using the auto keyword from C++11
#ifdef __CUDACC__
	gridtools::computation* copy =
#else
	    boost::shared_ptr<gridtools::computation> copy =
#endif
	    gridtools::make_computation<gridtools::BACKEND, layout_t>
	    (
		gridtools::make_mss // mss_descriptor
		(
		    execute<forward>(),
		    gridtools::make_esf<copy_functor>(p_in() // esf_descriptor
			)
		    ),
		domain, coords
		);

	copy->ready();

	copy->steady();

	copy->run();

// #define NX 5
// #define NY 5
// #define NZ 5

	std::vector<integrator_type::original_storage::pointer_type::pointee_t*> vec={in.fields()[0].get(), in.fields()[1].get()};
	he.pack(vec);

	he.exchange();

	he.unpack(vec);

        in.print();

	// in.print_value(NX,NY,0);
	// in.print_value(NX,0,NZ);
	// in.print_value(0,NY,NZ);
	// in.print_value(NX,NY,NZ);

	copy->finalize();

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return true;
    }

}//namespace copy_stencil
