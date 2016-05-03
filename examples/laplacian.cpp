#include "gtest/gtest.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <gridtools.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include <tools/verifier.hpp>
#include "Options.hpp"

/*! @file
  @brief  This file shows an implementation of the "laplace" stencil, similar to the one used in COSMO
*/
// [namespaces]
using namespace gridtools;
// [namespaces]

// [intervals]
/*!
  @brief This is the definition of the special regions in the "vertical" direction for the laplacian functor
  @tparam level is a struct containing two integers (a splitter and an offset) which identify a point on the vertical axis
*/
typedef gridtools::interval<level<0,-1>, level<1,-1> > x_lap;
/*!
  @brief This is the definition of the whole vertical axis
*/
typedef gridtools::interval<level<0,-2>, level<1,3> > axis;

// [intervals]
// [functor]
/**
   @brief structure containing the Laplacian-specific information.

Contains the stencil operators that compose the multistage stencil in this test
*/
struct lap_function {
    static const int n_args = 2; //!< public compile-time constant, \todo apparently useless?

    /**
       @brief placeholder for the output field, index 0. accessor contains a vector of 3 offsets and defines a plus method summing values to the offsets
    */
    typedef accessor<0, enumtype::inout, extent<>, 3 > out;
/**
       @brief  placeholder for the input field, index 1
    */
    typedef accessor<1, enumtype::in, extent<-1, 1, -1, 1>, 3 > in;
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
             dom(in(-1, 0, 0)) + dom(in( 0,-1, 0)));

    }
};

// [functor]

/*!
 * @brief This operator is used for debugging only
 */

std::ostream& operator<<(std::ostream& s, lap_function const) {
    return s << "lap_function";
}


// [start_main]
    /// \brief  Main function
    /// \param  argc An integer argument count of the command line arguments
    /// \param  argv An argument vector of the command line arguments
    /// \return an integer 0 upon exit success
TEST(Laplace, test) {

    uint_t d1 = Options::getInstance().m_size[0];
    uint_t d2 = Options::getInstance().m_size[1];
    uint_t d3 = Options::getInstance().m_size[2];

    uint_t halo_size=2;

    using namespace gridtools;
    using namespace enumtype;
// [start_main]

// [backend]
#ifdef CUDA_EXAMPLE
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif
// [backend]

// [layout_map]
    typedef gridtools::layout_map<0,1,2> layout_t;
// [layout_map]

// [storage_type]
    /**
       - definition of the storage type, depending on the BACKEND which is set as a macro. \todo find another strategy for the backend (policy pattern)?
    */
    typedef BACKEND::storage_info<0, layout_t> storage_info_t;
    typedef BACKEND::storage_type<float_type, storage_info_t >::type storage_type;
// [storage_type]

// [storage_initialization]
    /**
        - Instantiation of the actual data fields that are used for input/output
    */
    storage_info_t metadata_(d1,d2,d3);
    storage_type in(metadata_, -1., "in");
    storage_type out(metadata_, -7.3, "out");
// [storage_initialization]

// [placeholders]
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
    typedef boost::mpl::vector<p_in, p_out> accessor_list;
// [placeholders]

// [domain_type]
    /**
       - Construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
       It must be noted that the only fields to be passed to the constructor are the non-temporary.
       The order in which they have to be passed is the order in which they appear scanning the placeholders in order (i.e. the order in the accessor_list?). \todo (I don't particularly like this).
       \note domain_type implements the CRTP pattern in order to do static polymorphism (?) Because all what is 'clonable to gpu' must derive from the CRTP base class.
    */
       gridtools::domain_type<accessor_list> domain
        (boost::fusion::make_vector(&in, &out));
// [domain_type]

// [grid]
       /**
          - Definition of the physical dimensions of the problem.
          The grid constructor takes the horizontal plane dimensions,
          while the vertical ones are set according the the axis property soon after
       */
       uint_t di[5] = {halo_size, halo_size, halo_size, d1-halo_size, d1};
       uint_t dj[5] = {halo_size, halo_size, halo_size, d2-halo_size, d2};

       gridtools::grid<axis> grid(di,dj);
       grid.value_list[0] = 0;
       grid.value_list[1] = d3;
// [grid]

// [computation]
       /*!
         - Here we do lot of stuff:

         1) We pass to the intermediate representation ::run function the description
         of the stencil, which is a multi-stage stencil (mss)
         The mss includes (in order of execution) a laplacian, two fluxes which are independent
         and a final step that is the out_function

         2) The logical physical domain with the fields to use

         3) The actual domain dimensions

         \note in reality this call does nothing at runtime (besides assigning the runtime variables domain and grid), it only calls the constructor of the intermediate struct which is empty. the work done at compile time is documented in the \ref gridtools::intermediate "intermediate" class.
         \todo why is this function even called? It just needs to be compiled, in order to get the return type (use a typedef).
       */
#ifdef CXX11_ENABLED
       auto
#else
#ifdef __CUDACC__
       computation*
#else
       boost::shared_ptr<gridtools::computation>
#endif
#endif
       laplace = make_computation<gridtools::BACKEND>
        (
         domain, grid,
         make_mss //! \todo all the arguments in the call to make_mss are actually dummy.
         (
          execute<forward>(),//!\todo parameter used only for overloading purpose?
          make_esf<lap_function>(p_out(), p_in())//!  \todo elementary stencil function, also here the arguments are dummy.
          )
         );
// [computation]

// [ready_steady_run_finalize]
/**
   @brief This method allocates on the heap the temporary variables
   this method calls heap_allocated_temps::prepare_temporaries(...). It allocates the memory for the list of extents defined in the temporary placeholders (none).
 */
    laplace->ready();

/**
   @brief calls setup_computation and creates the local domains
   the constructors of the local domains get called (\ref gridtools::intermediate::instantiate_local_domain, which only initializes the dom public pointer variable)
   @note the local domains are allocated in the public scope of the \ref gridtools::intermediate struct, only the pointer is passed to the instantiate_local_domain struct
 */
    laplace->steady();

/**
   Call to gridtools::intermediate::run, which calls Backend::run, does the actual stencil operations on the backend.
 */
    laplace->run();

    laplace->finalize();

// [ready_steady_run_finalize]

    // [generate reference]

    storage_type ref(metadata_, -1., "ref");

    for(size_t i=2; i != d1-2; ++i) {
        for(size_t j=2; j != d2-2; ++j) {
            for(size_t k=0; k != d3; ++k) {
                ref(i,j,k) = 4*in(i,j,k) -
                        (in(i+1,j,k) + in(i,j+1,k) +
                         in(i-1, j, k) + in(i,j-1,k));
            }
        }
    }

#ifdef CXX11_ENABLED
    verifier verif(1e-13);
    array<array<uint_t, 2>, 3> halos{{ {halo_size,halo_size}, {halo_size,halo_size}, {halo_size,halo_size} }};
    bool result = verif.verify(grid, out, ref, halos);
#else
    verifier verif(1e-13, halo_size);
    bool result = verif.verify(grid, out, ref);
#endif

#ifdef BENCHMARK
        std::cout << laplace->print_meter() << std::endl;
#endif

    ASSERT_TRUE(result);
}

int main(int argc, char** argv)
{
    // Pass command line arguments to googltest
    ::testing::InitGoogleTest(&argc, argv);

    if (argc != 4) {
        printf( "Usage: laplace_<whatever> dimx dimy dimz\n where args are integer sizes of the data fields\n" );
        return 1;
    }

    for(int i=0; i!=3; ++i) {
        Options::getInstance().m_size[i] = atoi(argv[i+1]);
    }

    return RUN_ALL_TESTS();
}


/**
@}
*/
