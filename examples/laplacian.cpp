#include <stdio.h>
#include <stdlib.h>

#include <gridtools.h>
#ifdef CUDA_EXAMPLE
#include <stencil-composition/backend_cuda.h>
#else
#include <stencil-composition/backend_naive.h>
#endif

#include <boost/timer/timer.hpp>

/*! @file
  @brief  This file shows an implementation of the "horizontal diffusion" stencil, similar to the one used in COSMO

  Structure of the example:

  There are quantities computed at compile-time, and others which are computed at run-time.
  - Compile-time:
  -# The multi-stage stencil (MSS), it consists in the set of stencil operators which have to be executed sequentially at each iteration of the model.
  These stencil operations are implemented as functors, and define the Elementary Stencil Functions (ESF). We can thus assume that MSS is a vector of ESFs.
  -# The Elementary Stencil Function (ESF) is a functor defining one operator (e.g. differential operator, the Laplacian in this case). \
  It implements a templated method "Do" which performs the actual stencil operation.
  -# arg_type
  - Run-time
  -# The fields ("in" and "out" in this case) contain the values of a field on the grid. They live in the scope of the main function,
  their pointers are passed when the domain is constructed
  -# The global domain consists basically in the above mentioned fields ( accessed via a vector of pointers), and implements strategies to possibly
  copy to/from the backend
  -# The coordinates consist in the bounds for the loop over the spatial dimensions. They are stored in a struct which also implements a strategy
  to possibly copy them to/from the backend.
*/

using gridtools::level;
using gridtools::arg_type;
using gridtools::range;
using gridtools::arg;


/**
   @{
*/
/*!
  @brief This is the definition of the special regions in the "vertical" direction for the laplacian functor
  @tparam level is a struct containing two integers (a splitter and an offset) which identify a point on the vertical axis
*/
typedef gridtools::interval<level<0,-1>, level<1,-1> > x_lap;
/*!
  @brief The same for the flux functor
*/
typedef gridtools::interval<level<0,-1>, level<1,-1> > x_flx;
/*!
  @brief The same for the output functor
*/
typedef gridtools::interval<level<0,-1>, level<1,-1> > x_out;
/*!
  @brief This is the definition of the whole vertical axis
*/
typedef gridtools::interval<level<0,-2>, level<1,3> > axis;
/**
   @}
*/

/**
   @brief structure containing the Laplacian-specific information.

Contains the stencil operators that compose the multistage stencil in this test
*/
struct lap_function {
    static const int n_args = 2; //!< public compile-time constant, \todo apparently useless?

  /**
     @brief placeholder for the output field, index 0. arg_type contains a vector of 3 offsets and defines a plus method summing values to the offsets
  */
    typedef arg_type<0> out;
    /**
       @brief  placeholder for the input field, index 1
    */
    typedef const arg_type<1, range<-1, 1, -1, 1> > in;
    /**
       @brief MPL vector of the out and in types
    */
    typedef boost::mpl::vector<out, in> arg_list;

    /**
       @brief Do method, overloaded. t_domain specifies the policy, x_lapl is a dummy argument here \todo should it be removed?
    */
    template <typename t_domain>
    GT_FUNCTION
    static void Do(t_domain const & dom, x_lap) {

        dom(out()) = 4*dom(in()) -
            (dom(in( 1, 0, 0)) + dom(in( 0, 1, 0)) +
             dom(in(-1, 0, 0)) + dom(in( 0,-1, 0)));//! use of the arg_type<I> constructor in the Do function \todo why isn't the sum done at compile-time, e.g. expression templates?

    }
};

/**
@{
*/

/*!
 * @brief This operator is used for debugging only
 */

std::ostream& operator<<(std::ostream& s, lap_function const) {
    return s << "lap_function";
}


    /// \brief  Main function
    /// \param  argc An integer argument count of the command line arguments
    /// \param  argv An argument vector of the command line arguments
    /// \return an integer 0 upon exit success
int main(int argc, char** argv) {

    if (argc != 4) {
        std::cout << "Usage: interface1_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields" << std::endl;
        return 1;
    }

    /**
	The following steps are performed:

	- Definition of the domain:
    */
    int d1 = atoi(argv[1]); /** d1 cells in the x direction (horizontal)*/
    int d2 = atoi(argv[2]); /** d2 cells in the y direction (horizontal)*/
    int d3 = atoi(argv[3]); /** d3 cells in the z direction (vertical)*/

    using namespace gridtools;
    using namespace enumtype;

#ifdef CUDA_EXAMPLE
#define BACKEND backend<Cuda, Block>
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, Block>
#else
#define BACKEND backend<Host, Naive>
#endif
#endif

    /**
	- definition of the storage type, depending on the BACKEND which is set as a macro. \todo find another strategy for the backend (policy pattern)?
    */
    typedef gridtools::BACKEND::storage_type<double, gridtools::layout_map<0,1,2> >::type storage_type;
    /**
    - definition of the temporary storage type, also depends on the backend
	\todo unused here?
    */
    typedef gridtools::BACKEND::temporary_storage_type<double, gridtools::layout_map<0,1,2> >::type tmp_storage_type;

    /**
        - Instantiation of the actual data fields that are used for input/output
    */
    storage_type in(d1,d2,d3,-1, std::string("in"));
    storage_type out(d1,d2,d3,-7.3, std::string("out"));
    out.print();

    /**
       - Definition of placeholders. The order of them reflect the order the user will deal with them
       especially the non-temporary ones, in the construction of the domain.
       A placeholder only contains a static const index and a storage type
    */
    typedef arg<0, storage_type > p_in;
    typedef arg<1, storage_type > p_out;

    /**
       - Creation of an array of placeholders to be passed to the domain
       \todo I'm using mpl::vector, but the final API should look slightly simpler
    */
    typedef boost::mpl::vector<p_in, p_out> arg_type_list;

    /**
       - Construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
       It must be noted that the only fields to be passed to the constructor are the non-temporary.
       The order in which they have to be passed is the order in which they appear scanning the placeholders in order (i.e. the order in the arg_type_list?). \todo (I don't particularly like this).
       \note domain_type implements the CRTP pattern in order to do static polymorphism (?) Because all what is 'clonable to gpu' must derive from the CRTP base class.
    */
       gridtools::domain_type<arg_type_list> domain
        (boost::fusion::make_vector(&in, &out));

       /**
	  - Definition of the physical dimensions of the problem.
	  The coordinates constructor takes the horizontal plane dimensions,
	  while the vertical ones are set according the the axis property soon after
       */
       gridtools::coordinates<axis> coords(2,d1-2,2,d2-2);
       coords.value_list[0] = 0;
       coords.value_list[1] = d3;

    /*!
      - Here we do lot of stuff:

      1) We pass to the intermediate representation ::run function the description
      of the stencil, which is a multi-stage stencil (mss)
      The mss includes (in order of execution) a laplacian, two fluxes which are independent
      and a final step that is the out_function

      2) The logical physical domain with the fields to use

      3) The actual domain dimensions

      \note in reality this call does nothing at runtime (besides assigning the runtime variables domain and coords), it only calls the constructor of the intermediate struct which is empty. the work done at compile time is documented in the \ref gridtools::intermediate "intermediate" class.
      \todo why is this function even called? It just needs to be compiled, in order to get the return type (use a typedef).
     */

#ifdef __CUDACC__
    gridtools::computation* horizontal_diffusion =
#else
    boost::shared_ptr<gridtools::computation> horizontal_diffusion =
#endif
        gridtools::make_computation<gridtools::BACKEND>
        (
         gridtools::make_mss //! \todo all the arguments in the call to make_mss are actually dummy.
         (
          gridtools::execute_upward,//!\todo parameter used only for overloading purpose?
          gridtools::make_esf<lap_function>(p_out(), p_in())//!  \todo elementary stencil function, also here the arguments are dummy.
          ),
         domain, coords);

/**
   @brief This method allocates on the heap the temporary variables
   this method calls heap_allocated_temps::prepare_temporaries(...). It allocates the memory for the list of ranges defined in the temporary placeholders (none).
 */
    horizontal_diffusion->ready();

    domain.storage_info<boost::mpl::int_<0> >();
    domain.storage_info<boost::mpl::int_<1> >();

/**
   @brief calls setup_computation and creates the local domains
   the constructors of the local domains get called (\ref gridtools::intermediate::instantiate_local_domain, which only initializes the dom public pointer variable)
   @note the local domains are allocated in the public scope of the \ref gridtools::intermediate struct, only the pointer is passed to the instantiate_local_domain struct
 */
    horizontal_diffusion->steady();
    printf("\n\n\n\n\n\n");
/**
   @brief in case GPUs are used calls a kernel on the GPU allocating space (with new)
*/
    domain.clone_to_gpu();
    printf("CLONED\n");

    boost::timer::cpu_timer time;
/**
   Call to gridtools::intermediate::run, which calls Backend::run, does the actual stencil operations on the backend.
 */
    horizontal_diffusion->run();
    boost::timer::cpu_times lapse_time = time.elapsed();

    horizontal_diffusion->finalize();

#ifdef CUDA_EXAMPLE
    out.data.update_cpu();
#endif

    //    in.print();
    out.print();
    //    lap.print();

    std::cout << "TIME " << boost::timer::format(lapse_time) << std::endl;

     return 0;
}

/**
@}
*/
