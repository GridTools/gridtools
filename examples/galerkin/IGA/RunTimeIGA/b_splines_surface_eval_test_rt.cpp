// Non GT headers
#include "b_splines_surface_rt.h"
#include "point.h"
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
constexpr int N(7); // Number of base functions
constexpr int D(2); // Surface space dimension
constexpr int NK(P+N+1);// TODO: check this
constexpr std::array<double, NK> knots {0,0,0,1,2,3,4,5,5,5};


std::vector<iga_rt::Point<D> > control_points(N);


// Non-GT type b-spline instance allocation
iga_rt::BSplineCurve<P,N,D> curve(knots, control_points);


////////////////// GT-STYLE CODE PART /////////////////////

struct curve_struct
{
    static const int n_args = 2;

    typedef accessor<0, gridtools::enumtype::inout, extent<0, 0, 0, 0>, 4> curve_values;

    typedef const accessor<1, gridtools::enumtype::in, extent<0, 0, 0, 0>, 3 > csi;

    typedef boost::mpl::vector<curve_values, csi> arg_list;

    template <typename t_domain>
    GT_FUNCTION
    static void Do(t_domain const & dom, x_lap) {

    	const iga_rt::Point<D> curveValue(curve.evaluate(dom(csi())));

    	// Loop over point coordinates
    	for(int coords=0;coords<D;++coords)
    	{
    		dom(curve_values(dimension<4>(coords))) = curveValue.m_coords[coords];
    	}
    }

};

int main()
{

	////////////////// NON GT-STYLE CODE PART /////////////////////


	control_points[0].m_coords[0] = 0.;
	control_points[0].m_coords[1] = 1.;
	control_points[1].m_coords[0] = 0.5;
	control_points[1].m_coords[1] = 3.;
	control_points[2].m_coords[0] = 1.5;
	control_points[2].m_coords[1] = 3.;
	control_points[3].m_coords[0] = 2.5;
	control_points[3].m_coords[1] = 0.;
	control_points[4].m_coords[0] = 3.5;
	control_points[4].m_coords[1] = 2.;
	control_points[5].m_coords[0] = 4.5;
	control_points[5].m_coords[1] = 1.;
	control_points[6].m_coords[0] = 5.;
	control_points[6].m_coords[1] = 4.;


	// Non-GT style b-spline evaluation point preparation
	constexpr double minCsi(0.);
	constexpr double maxCsi(5.);
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

    typedef gridtools::BACKEND::storage_info<__COUNTER__, layout_t_out> storage_type_curve_basis_info;
    typedef gridtools::BACKEND::storage_type<gridtools::float_type
                                             , storage_type_curve_basis_info>::type storage_type_curve_basis_values;

    // Storage container allocation
    storage_type_csi_info csi_info(numPoints,1,1);
    storage_type_csi csi(csi_info);
    storage_type_curve_basis_info bspline_basis_info(numPoints,1,1,N);
    storage_type_curve_basis_values bspline_basis_values(bspline_basis_info);

    // Fill csi-storage
    for(unsigned int csiIndex=0;csiIndex<numPoints;++csiIndex)
	{
    	csi(csiIndex,0,0) = csiValues[csiIndex];
	}


    // Definition of csi and bspline values storage placeholders
    typedef gridtools::arg<0,storage_type_csi> p_csi;
    typedef gridtools::arg<1,storage_type_curve_basis_values> p_curve_values;
    typedef boost::mpl::vector<p_csi,p_curve_values> placeholder_list;

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
    computation* curve_calculation =
#else
    boost::shared_ptr<gridtools::computation> curve_calculation =
#endif
    		gridtools::make_computation<gridtools::BACKEND>(gridtools::make_multistage(gridtools::enumtype::execute<gridtools::enumtype::forward>(),
    																	 gridtools::make_stage<curve_struct>(p_curve_values(),
    		  	  	  	  	  	  	  	  	  	  	  	  	  			 	 	 	 	 	    		   p_csi())),
																		 domain,
																		 coordinates);

    // Computation execution
    curve_calculation->ready();
    curve_calculation->steady();
    curve_calculation->run();
    curve_calculation->finalize();


    ////////////////// NON GT-STYLE CODE PART /////////////////////

    // Results check with standard non GT calculation
    iga_rt::Point<D> curveValue;
    for(int csiIndex=0;csiIndex<numPoints;++csiIndex)
	{
    	curveValue = curve.evaluate(csiValues[csiIndex]);

		for(int coords=0;coords<D;++coords)
		{
	    	if(curveValue.m_coords[coords]!=bspline_basis_values(csiIndex,0,0,coords))
			{
                            std::cout<<bspline_basis_values(csiIndex,0,0,coords)<<" "<<curveValue.m_coords[coords]<<std::endl;
			}
		}
	}


	return 0;

}
