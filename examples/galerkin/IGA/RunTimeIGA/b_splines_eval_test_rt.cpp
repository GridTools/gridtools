// Non GT headers
#include "b_splines_rt.h"
#include <vector>
#include <iostream>
#include <boost/mpl/vector.hpp>
#include <boost/shared_ptr.hpp>

// GT headers and namespaces
#include <stencil-composition/backend.hpp>
#include <stencil-composition/range.hpp>
#include <stencil-composition/level.hpp>
#include <stencil-composition/accessor.hpp>
#include <common/defs.hpp>
#include <stencil-composition/make_computation.hpp>
using gridtools::range;
using gridtools::level;
using gridtools::interval;
using gridtools::accessor;


typedef interval<level<0,-1>, level<1,-1> > x_lap;
typedef interval<level<0,-1>, level<1,1> > third_axis;

////////////////// NON GT-STYLE CODE PART /////////////////////

// Non-GT type b-spline node set creation
constexpr int P(2); // b-spline order
constexpr int N(5); // Number of function
constexpr int NK(P+N+1);// TODO: check this
constexpr double knots[NK] = {0,1,2,3,4,5,6};

// Non-GT type b-spline instance allocation
constexpr int I(2); // b-spline index
b_splines_rt::BSpline<I,P> bspline(knots);


////////////////// GT-STYLE CODE PART /////////////////////


struct bspline_struct
{
    static const int n_args = 2;

    typedef accessor<0, range<0, 0, 0, 0>, 3 > bsline_values;

    typedef const accessor<1, range<0, 0, 0, 0>, 3 > csi;

    typedef boost::mpl::vector<bsline_values, csi> arg_list;

//    typedef mybasisfunctions::BSpline<I,P>* bspline_ptr;

    template <typename t_domain>
    GT_FUNCTION
    static void Do(t_domain const & dom, x_lap) {

        dom(bsline_values()) = bspline.evaluate(dom(csi()));

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
    typedef gridtools::layout_map<0,1,2> layout_t_out;

    // Kernel backend definition
    #ifdef CUDA_EXAMPLE
    #define BACKEND backend<gridtools::enumtype::Cuda,gridtools::enumtype:Block>
    #else
    #ifdef BACKEND_BLOCK
    #define BACKEND backend<gridtools::enumtype::Host,gridtools::enumtype::Block>
    #else
    #define BACKEND backend<gridtools::enumtype::Host,gridtools::enumtype::Naive>
    #endif
    #endif

    // Storage type definition
    typedef gridtools::storage_info<layout_t_in, __COUNTER__> storage_type_csi_info;
    typedef gridtools::BACKEND::storage_type<gridtools::float_type
                                             , storage_type_csi_info>::type storage_type_csi;

    typedef gridtools::storage_info<layout_t_out, __COUNTER__> storage_type_bspline_basis_info;
    typedef gridtools::BACKEND::storage_type<gridtools::float_type
                                             , storage_type_bspline_basis_info>::type storage_type_bspline_basis_values;

    // Storage container allocation (sharing the same storage info)
    storage_type_csi_info csi_info(numPoints,1,1);
    storage_type_csi csi(csi_info);
    storage_type_bspline_basis_values bspline_basis_values(csi_info);


    // Fill csi-storage
    for(unsigned int csiIndex=0;csiIndex<numPoints;++csiIndex)
	{
    	csi(csiIndex,0,0) = csiValues[csiIndex];
	}


    // Definition of csi and bspline values storage placeholders
    typedef gridtools::arg<0,storage_type_csi> p_csi;
    typedef gridtools::arg<1,storage_type_bspline_values> p_bspline_values;
    typedef boost::mpl::vector<p_csi,p_bspline_values> placeholder_list;

    // Domain (data) definition, here the correspondence between place holders and real containers is being created:
    // placeholder_list contains the set of placeholders corresponding to the real containers provided to the
    // make_vector function (same order!). It must be noted that the same strorage_type is used for real container
    // instances and placeholder templated argument. In their turns these storage type contains the infos about the
    // stored data type (e.g., float, etc) and memory layout
    gridtools::domain_type<placeholder_list> domain(boost::fusion::make_vector(&csi, &bspline_values));

    // Domain (coordinates structure+halos) definition
    gridtools::uint_t csi_indexes[5] = {0, 0, 0, numPoints-1, numPoints}; // Knot (csi) domain direction definition
    gridtools::uint_t null_indexes[5] = {0, 0, 0, 0, 1}; // Second domain direction not required
    gridtools::coordinates<third_axis> coordinates(csi_indexes,null_indexes);
    coordinates.value_list[0] = 0; // Third (vertical direction) index structure (start)
    coordinates.value_list[1] = 0; // Third (vertical direction) index structure (stop): it must be noted that in this direction the loop stop condition has a "<="

    // Compuation esf and mss definition
#ifdef __CUDACC__
    computation* bspline_calculation =
#else
    boost::shared_ptr<gridtools::computation> bspline_calculation =
#endif
    		gridtools::make_computation<gridtools::BACKEND, layout_t_in>(gridtools::make_mss(gridtools::enumtype::execute<gridtools::enumtype::forward>(),
    																	 gridtools::make_esf<bspline_struct>(p_bspline_values(),
    		  	  	  	  	  	  	  	  	  	  	  	  	  			 	 	 	 	 	    		     p_csi())),
																		 domain,
																		 coordinates);

    // Computation execution
    bspline_calculation->ready();
    bspline_calculation->steady();
    bspline_calculation->run();
    bspline_calculation->finalize();


    ////////////////// NON GT-STYLE CODE PART /////////////////////

    // Results check with standard non GT calculation
    for(unsigned int csiIndex=0;csiIndex<numPoints;++csiIndex)
	{
		if(bspline.evaluate(csiValues[csiIndex])!=bspline_values(csiIndex,0,0))
		{
			std::cout<<bspline.evaluate(csiValues[csiIndex])<<" "<<bspline_values(csiIndex,0,0)<<std::endl;
		}
	}


	return 0;

}
