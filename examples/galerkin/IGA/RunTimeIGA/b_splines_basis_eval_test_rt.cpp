// Non GT headers
#include "b_splines_basis_rt.h"
#include <vector>
#include <iostream>
#include <boost/mpl/vector.hpp>
#include <boost/shared_ptr.hpp>

// GT headers and namespaces
#include <stencil-composition/stencil-composition.hpp>
using gridtools::extent;
using gridtools::level;
using gridtools::interval;
using gridtools::accessor;
using gridtools::dimension;


typedef interval<level<0,-1>, level<1,-1> > x_lap;
typedef interval<level<0,-1>, level<1,1> > third_axis;



////////////////// NON GT-STYLE CODE PART /////////////////////

// Non-GT type b-spline node set creation
constexpr int P(2); // b-spline order
constexpr int N(5); // Number of function
constexpr int NK(P+N+1);// TODO: check this
constexpr const std::array<double, NK> knots{0,1,2,3,4,5,6,7};

// Non-GT type b-spline instance allocation
iga_rt::BSplineBasis<P,N> bspline_basis(knots);


////////////////// GT-STYLE CODE PART /////////////////////

struct bspline_basis_struct
{
    static const int n_args = 2;

    typedef accessor<0, gridtools::enumtype::inout, extent<0, 0, 0, 0>, 4 > bspline_basis_values;

    typedef const accessor<1, gridtools::enumtype::in, extent<0, 0, 0, 0>, 3 > csi;

    typedef boost::mpl::vector<bspline_basis_values, csi> arg_list;

    template <typename t_domain>
    GT_FUNCTION
    static void Do(t_domain const & dom, x_lap) {

    	const std::vector<double> basis_function_values(bspline_basis.evaluate(dom(csi())));

    	for(int basis_index=0;basis_index<N;++basis_index)
    	{
    		dom(bspline_basis_values(dimension<4>(basis_index))) = basis_function_values[basis_index];
    	}
    }

};

int main()
{

    ////////////////// NON GT-STYLE CODE PART /////////////////////

    // Non-GT style b-spline evaluation point preparation
    constexpr double minCsi(0.);
    constexpr double maxCsi(6.);
    constexpr int numPoints(1000);
    constexpr double deltaCsi =(maxCsi-minCsi)/numPoints;
    std::vector<double> csiValues(numPoints);
    double currentCsiValue=minCsi;
    for(unsigned int csiIndex=0;csiIndex<numPoints;++csiIndex,currentCsiValue+=deltaCsi)
    {
        csiValues[csiIndex] = currentCsiValue;
    }

    ////////////////// GT-STYLE CODE PART /////////////////////

    // Memory layout definition
    typedef gridtools::layout_map<0,1,2> layout_t_in;
    typedef gridtools::layout_map<0,1,2,3> layout_t_out;

    // Kernel backend definition
    #ifdef CUDA_EXAMPLE
    #define BACKEND backend<gridtools::enumtype::Cuda, GRIDBACKEND, gridtools::enumtype:Block>
    #else
    #ifdef BACKEND_BLOCK
    #define BACKEND backend<gridtools::enumtype::Host, GRIDBACKEND, gridtools::enumtype::Block>
    #else
    #define BACKEND backend<gridtools::enumtype::Host, GRIDBACKEND, gridtools::enumtype::Naive>
    #endif
    #endif

    // Storage type definition
    typedef gridtools::BACKEND::storage_info< __COUNTER__, layout_t_in> storage_type_csi_info;
    typedef gridtools::BACKEND::storage_type<gridtools::float_type
                                             , storage_type_csi_info>::type storage_type_csi;

    typedef gridtools::BACKEND::storage_info< __COUNTER__, layout_t_out> storage_type_bspline_basis_info;
    typedef gridtools::BACKEND::storage_type<gridtools::float_type
                                             , storage_type_bspline_basis_info>::type storage_type_bspline_basis_values;

    // Storage container allocation
    storage_type_csi_info csi_info(numPoints,1,1);
    storage_type_csi csi(csi_info);
    storage_type_bspline_basis_info bspline_basis_info(numPoints,1,1,N);
    storage_type_bspline_basis_values bspline_basis_values(bspline_basis_info);

    // Fill csi-storage
    for(unsigned int csiIndex=0;csiIndex<numPoints;++csiIndex)
	{
    	csi(csiIndex,0,0) = csiValues[csiIndex];
	}


    // Definition of csi and bspline values storage placeholders
    typedef gridtools::arg<0,storage_type_csi> p_csi;
    typedef gridtools::arg<1,storage_type_bspline_basis_values> p_bspline_basis_values;
    typedef boost::mpl::vector<p_csi,p_bspline_basis_values> placeholder_list;

    // Domain (data) definition, here the correspondence between place holders and real containers is being created:
    // placeholder_list contains the set of placeholders corresponding to the real containers provided to the
    // make_vector function (same order!). It must be noted that the same strorage_type is used for real container
    // instances and placeholder templated argument. In their turns these storage type contains the infos about the
    // stored data type (e.g., float, etc) and memory layout
    gridtools::aggregator_type<placeholder_list> domain(boost::fusion::make_vector(&csi, &bspline_basis_values));

    // Domain (coordinates structure+halos) definition
    gridtools::uint_t csi_indexes[5] = {0, 0, 0, numPoints-1, numPoints}; // Knot (csi) domain direction definition
    gridtools::uint_t null_indexes[5] = {0, 0, 0, 0, 1}; // Second domain direction not required
    gridtools::grid<third_axis> coordinates(csi_indexes,null_indexes);
    coordinates.value_list[0] = 0; // Third (vertical direction) index structure (start)
    coordinates.value_list[1] = 0; // Third (vertical direction) index structure (stop): it must be noted that in this direction the loop stop condition has a "<="

    // Compuation esf and mss definition
#ifdef __CUDACC__
    computation* bspline_basis_calculation =
#else
    boost::shared_ptr<gridtools::computation> bspline_basis_calculation =
#endif
    		gridtools::make_computation<gridtools::BACKEND>(gridtools::make_multistage(gridtools::enumtype::execute<gridtools::enumtype::forward>(),
    																	 gridtools::make_stage<bspline_basis_struct>(p_bspline_basis_values(),
    		  	  	  	  	  	  	  	  	  	  	  	  	  			 	 	 	 	 	    		     p_csi())),
																		 domain,
																		 coordinates);

    // Computation execution
    bspline_basis_calculation->ready();
    bspline_basis_calculation->steady();
    bspline_basis_calculation->run();
    bspline_basis_calculation->finalize();


    ////////////////// NON GT-STYLE CODE PART /////////////////////

    // Results check with standard non GT calculation
    std::vector<double> basis_function_values(N);
    for(int csiIndex=0;csiIndex<numPoints;++csiIndex)
	{
    	basis_function_values = bspline_basis.evaluate(csiValues[csiIndex]);
    	for(int basis_index=0;basis_index<N;++basis_index)
    	{
    		if(basis_function_values[basis_index]!=bspline_basis_values(csiIndex,0,0,basis_index))
    		{
    			std::cout<<basis_function_values[basis_index]<<" "<<bspline_basis_values(csiIndex,0,0,basis_index)<<std::endl;
    		}
    	}
	}


	return 0;

}
