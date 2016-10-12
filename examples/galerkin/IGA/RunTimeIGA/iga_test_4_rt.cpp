#include "b_splines_surface_rt.h"
#include "nurbs_surface_rt.h"
#include "b_splines_rt.h"
#include "b_splines_basis_rt.h"
#include "nd_b_splines_basis_rt.h"
#include "nurbs_basis_rt.h"
#include "nd_nurbs_basis_rt.h"
#include "point.h"
#include "common/array.hpp"
#include <array>
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>

#define _USE_MATH_DEFINES
#include <cmath>
#include <iomanip>

int main()
{
	constexpr int P1(1);
	constexpr int N1(2);
	constexpr int P1p(2);
	constexpr int N1p(3);
	constexpr int P2(2);
	constexpr int N2(4);
	constexpr int DIM(2);
	constexpr gridtools::array<double,N1> weights1{1.,1.};
	constexpr gridtools::array<double,N2> weights2{1.,1.,1.,1.};
	constexpr gridtools::array<double,iga_rt::ComputeMultiplicity<P1,N1,P2,N2>::m_multiplicity> weights{1.,1.,1.,1.,1.,1.,1.,1.};
	constexpr gridtools::array<double,4> knots1{0.,0.,1.,1.};
	constexpr gridtools::array<double,6> knots1p{0.,0.,0.,1.,1.,1.};
	constexpr gridtools::array<double,7> knots2{0.,0.,0.,0.5,1.,1.,1.};
	constexpr gridtools::array<double,iga_rt::ComputeKnotSetSize<P1,N1,P2,N2>::m_size> knots{0.,0.,1.,1.,0.,0.,0.,0.5,1.,1.,1};
	constexpr std::array<double,N1> weights1_std{1.,1.};
	constexpr std::array<double,N2> weights2_std{1.,1.,1.,1.};
	constexpr std::array<double,iga_rt::ComputeMultiplicity<P1,N1,P2,N2>::m_multiplicity> weights_std{1.,1.,1.,1.,1.,1.,1.,1.};
	constexpr std::array<double,4> knots1_std{0.,0.,1.,1.};
	constexpr std::array<double,6> knots1p_std{0.,0.,0.,1.,1.,1.};
	constexpr std::array<double,7> knots2_std{0.,0.,0.,0.5,1.,1.,1.};
	constexpr std::array<double,iga_rt::ComputeKnotSetSize<P1,N1,P2,N2>::m_size> knots_std{0.,0.,1.,1.,0.,0.,0.,0.5,1.,1.,1};

	std::cout<<std::setprecision(20);
	std::default_random_engine generator1(1);
	std::uniform_real_distribution<double> distribution1(0.0,1.0);
	std::default_random_engine generator2(2);
	std::uniform_real_distribution<double> distribution2(0.0,1.0);


	iga_rt::NDBSplineBasis<P2,N2> nd_test1(knots2);
	iga_rt::BSplineBasis<P2,N2> test(knots2_std);

	for(unsigned int i=0;i<100;++i)
	{
		const double val(distribution1(generator1));
		for(int j=0;j<N2;++j)
		{
			if(nd_test1.evaluate(gridtools::array<double,1>{val})[j]!=test.evaluate(val)[j])
			{
				std::cout<<nd_test1.evaluate(gridtools::array<double,1>{val})[j]<<" "<<test.evaluate(val)[j]<<std::endl;
			}
			if(nd_test1.evaluateDerivatives(gridtools::array<double,1>{val})[j]!=test.evaluateDerivatives(val)[j])
			{
				std::cout<<"deriv "<<nd_test1.evaluateDerivatives(gridtools::array<double,1>{val})[j]<<" "<<test.evaluateDerivatives(val)[j]<<std::endl;
			}
		}
	}



	iga_rt::NDBSplineBasis<P1,N1,P2,N2> nd_test2(knots);
	iga_rt::BivariateBSplineBasis<P1,N1,P2,N2> bv_test(knots1_std,knots2_std);


	for(unsigned int i=0;i<100;++i)
	{
		const double val1(distribution1(generator1));
		const double val2(distribution1(generator2));
		int global_index(0);
		int global_index_der(0);


		for(int j1=0;j1<N1;++j1)
		{
			for(int j2=0;j2<N2;++j2,++global_index)
			{
				if(nd_test2.evaluate(gridtools::array<double,2>{val1,val2})[global_index]!=bv_test.evaluate(val1,val2)[global_index])
				{
					std::cout<<nd_test2.evaluate(gridtools::array<double,2>{val1,val2})[global_index]<<" "<<bv_test.evaluate(val1,val2)[global_index]<<std::endl;
				}
			}
		}

		for(int j1=0;j1<iga_rt::ComputeDimension<P1,N1,P2,N2>::m_dimension;++j1)
		{
			global_index = 0;
			for(int j2=0;j2<N1;++j2)
			{
				for(int j3=0;j3<N2;++j3,++global_index_der,++global_index)
				{
					if(nd_test2.evaluateDerivatives(gridtools::array<double,2>{val1,val2})[global_index_der]!=bv_test.evaluateDerivatives(val1,val2)[j1][global_index])
					{
						std::cout<<j1<<" "<<j2<<" "<<j3<<" "<<nd_test2.evaluateDerivatives(gridtools::array<double,2>{val1,val2})[global_index_der]<<" "<<bv_test.evaluateDerivatives(val1,val2)[j1][global_index]<<std::endl;
					}
				}

			}
		}
	}

	iga_rt::NurbsBasis<P2,N2> nurbs_test1(knots2_std,&(weights2_std[0]));
	iga_rt::NDNurbsBasis<P2,N2> nd_nurbs_test1(knots2,weights2);
	for(unsigned int i=0;i<100;++i)
	{
		const double val(distribution1(generator1));
		for(int j=0;j<N2;++j)
		{
//			std::cout<<nd_nurbs_test1.evaluate(gridtools::array<double,1>{val})[j]<<std::endl;
//			std::cout<<"deriv "<<nd_nurbs_test1.evaluateDerivatives(gridtools::array<double,1>{val})[j]<<std::endl;
			if(nd_nurbs_test1.evaluate(gridtools::array<double,1>{val})[j]!=nurbs_test1.evaluate(val)[j])
			{
				std::cout<<nd_nurbs_test1.evaluate(gridtools::array<double,1>{val})[j]<<" "<<nurbs_test1.evaluate(val)[j]<<std::endl;
			}
			if(nd_nurbs_test1.evaluateDerivatives(gridtools::array<double,1>{val})[j]!=nurbs_test1.evaluateDerivatives(val)[j])
			{
				std::cout<<"deriv "<<nd_nurbs_test1.evaluateDerivatives(gridtools::array<double,1>{val})[j]<<" "<<nurbs_test1.evaluateDerivatives(val)[j]<<std::endl;
			}
		}
	}




	iga_rt::NDNurbsBasis<P1,N1,P2,N2> nd_nurbs_test2(knots,weights);
	iga_rt::BivariateNurbsBasis<P1,N1,P2,N2> nurbs_test2(knots1_std,knots2_std,&(weights_std[0]));

	for(unsigned int i=0;i<100;++i)
	{
		const double val1(distribution1(generator1));
		const double val2(distribution1(generator2));
		int global_index(0);
		int global_index_der(0);


		for(int j1=0;j1<N1;++j1)
		{
			for(int j2=0;j2<N2;++j2,++global_index)
			{
//				std::cout<<nd_nurbs_test2.evaluate(gridtools::array<double,2>{val1,val2})[global_index]<<std::endl;
				if(nd_nurbs_test2.evaluate(gridtools::array<double,2>{val1,val2})[global_index]!=nurbs_test2.evaluate(val1,val2)[global_index])
				{
					std::cout<<nd_nurbs_test2.evaluate(gridtools::array<double,2>{val1,val2})[global_index]<<" "<<nurbs_test2.evaluate(val1,val2)[global_index]<<std::endl;
				}
			}
		}

		for(int j1=0;j1<iga_rt::ComputeDimension<P1,N1,P2,N2>::m_dimension;++j1)
		{
			global_index = 0;
			for(int j2=0;j2<N1;++j2)
			{
				for(int j3=0;j3<N2;++j3,++global_index_der,++global_index)
				{
//					std::cout<<j1<<" "<<j2<<" "<<j3<<" "<<nd_nurbs_test2.evaluateDerivatives(gridtools::array<double,2>{val1,val2})[global_index_der]<<std::endl;
					if(nd_nurbs_test2.evaluateDerivatives(gridtools::array<double,2>{val1,val2})[global_index_der]!=nurbs_test2.evaluateDerivatives(val1,val2)[j1][global_index])
					{
						std::cout<<j1<<" "<<j2<<" "<<j3<<" "<<nd_nurbs_test2.evaluateDerivatives(gridtools::array<double,2>{val1,val2})[global_index_der]<<" "<<nurbs_test2.evaluateDerivatives(val1,val2)[j1][global_index]<<std::endl;
					}
				}

			}
		}
	}


#if 0

//	constexpr double surface_weigths[] = {1.,(1.+1./std::sqrt(2))/2.,(1.+1./std::sqrt(2))/2.,1.,1.,(1.+1./std::sqrt(2))/2.,(1.+1./std::sqrt(2))/2.,1.};
	constexpr double surface_weigths[] = {1.,1.,1.,1.,1.,1.,1.,1.};

	std::vector<iga_rt::Point<DIM>> rainbow_control_points(8);
/*
	rainbow_control_points[0].m_coords[0] = 1.;
	rainbow_control_points[0].m_coords[1] = 0.;

	rainbow_control_points[1].m_coords[0] = 1.;
	rainbow_control_points[1].m_coords[1] = std::sqrt(2.) - 1.;

	rainbow_control_points[2].m_coords[0] = std::sqrt(2.) - 1.;
	rainbow_control_points[2].m_coords[1] = 1.;

	rainbow_control_points[3].m_coords[0] = 0.;
	rainbow_control_points[3].m_coords[1] = 1.;

	rainbow_control_points[4].m_coords[0] = 2.;
	rainbow_control_points[4].m_coords[1] = 0.;

	rainbow_control_points[5].m_coords[0] = 2.;
	rainbow_control_points[5].m_coords[1] = 2.*(std::sqrt(2.) - 1.);

	rainbow_control_points[6].m_coords[0] = 2.*(std::sqrt(2.) - 1.);
	rainbow_control_points[6].m_coords[1] = 2.;

	rainbow_control_points[7].m_coords[0] = 0.;
	rainbow_control_points[7].m_coords[1] = 2.;
*/

	rainbow_control_points[0].m_coords[0] = 0.;
	rainbow_control_points[0].m_coords[1] = 0.;

	rainbow_control_points[1].m_coords[0] = 0.;
	rainbow_control_points[1].m_coords[1] = 0.;

	rainbow_control_points[2].m_coords[0] = 1.;
	rainbow_control_points[2].m_coords[1] = 1.;

	rainbow_control_points[3].m_coords[0] = 0.;
	rainbow_control_points[3].m_coords[1] = 0.;

	rainbow_control_points[4].m_coords[0] = 0.;
	rainbow_control_points[4].m_coords[1] = 0.;

	rainbow_control_points[5].m_coords[0] = 0.;
	rainbow_control_points[5].m_coords[1] = 0.;

	rainbow_control_points[6].m_coords[0] = 0.;
	rainbow_control_points[6].m_coords[1] = 0.;

	rainbow_control_points[7].m_coords[0] = 0.;
	rainbow_control_points[7].m_coords[1] = 0.;


//	iga_rt::NurbsSurface<P1,N1,P2,N2,DIM> rainbow_surface(knots1, knots2, surface_weigths, rainbow_control_points);
	iga_rt::BSplineSurface<P1,N1,P2,N2,DIM> rainbow_surface(knots1, knots2, rainbow_control_points);
	iga_rt::BSplineBasis<P2,N2> second_basis(knots2);
	iga_rt::BivariateBSplineBasis<P1,N1,P2,N2> surf_basis(knots1,knots2);

	// Monte Carlo integration

	std::default_random_engine generator2(2);
	std::uniform_real_distribution<double> distribution2(0.0,1.0);
	std::default_random_engine generator1(1);
	std::uniform_real_distribution<double> distribution1(0.0,1.0);
	double rainbow_area(0.);
	double rainbow_area_squared(0.);
	for(unsigned int i=0;i<1;++i)
	{
//		const double gen1(distribution1(generator1));
//		const double gen2(distribution2(generator2));
		const double gen1(0.7);
		const double gen2(0.7);
		std::array<iga_rt::Point<DIM>,2> jacobianOld = rainbow_surface.evaluateJacobianOld(gen1,gen2);
		std::array<iga_rt::Point<DIM>,2> jacobian = rainbow_surface.evaluateJacobian(gen1,gen2);
//		std::array<iga_rt::Point<DIM>,2> jacobianOld = rainbow_surface.evaluateJacobianOld(0.,0.);
//		std::array<iga_rt::Point<DIM>,2> jacobian = rainbow_surface.evaluateJacobian(0.,0.);

		std::cout<<jacobianOld[0].m_coords[0]<<" "<<jacobianOld[0].m_coords[1]<<" "<<jacobianOld[1].m_coords[0]<<" "<<jacobianOld[1].m_coords[1]<<std::endl;
		std::cout<<jacobian[0].m_coords[0]<<" "<<jacobian[0].m_coords[1]<<" "<<jacobian[1].m_coords[0]<<" "<<jacobian[1].m_coords[1]<<std::endl;
		std::cout<<second_basis.evaluate(gen2)[2]<<" "<<surf_basis.evaluateDerivatives(gen1,gen2)[0][2]<<std::endl;
		std::cout<<std::endl;
	}

#endif

	return 0;
}
