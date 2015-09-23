#include "b_splines_surface_rt.h"
#include "nurbs_surface_rt.h"
#include "point.h"
#include <array>
#include <vector>
#include <iostream>
#include <random>

#define _USE_MATH_DEFINES
#include <cmath>

int main()
{
	// TODO: update all test main and switch to gtest

	constexpr int P1(1);
	constexpr int N1(2);
	constexpr int P1p(2);
	constexpr int N1p(3);
	constexpr int P2(2);
	constexpr int N2(4);
	constexpr int DIM(2);
	constexpr std::array<double,4> knots1{0.,0.,1.,1.};
	constexpr std::array<double,6> knots1p{0.,0.,0.,1.,1.,1.};
	constexpr std::array<double,7> knots2{0.,0.,0.,0.5,1.,1.,1.};


	// Test length calculation of unity circle line (a quarter of)

	constexpr double curve_weigths[] = {1.,(1.+1./std::sqrt(2))/2.,(1.+1./std::sqrt(2))/2.,1.};

	std::vector<iga_rt::Point<DIM>> rainbow_curve_control_points(N2);
	rainbow_curve_control_points[0].m_coords[0] = 1.;
	rainbow_curve_control_points[0].m_coords[1] = 0.;

	rainbow_curve_control_points[1].m_coords[0] = 1.;
	rainbow_curve_control_points[1].m_coords[1] = 1.*(std::sqrt(2.) - 1.);

	rainbow_curve_control_points[2].m_coords[0] = 1.*(std::sqrt(2.) - 1.);
	rainbow_curve_control_points[2].m_coords[1] = 1.;

	rainbow_curve_control_points[3].m_coords[0] = 0.;
	rainbow_curve_control_points[3].m_coords[1] = 1.;

	iga_rt::NurbsCurve<P2,N2,DIM> rainbow_curve(knots2, curve_weigths, rainbow_curve_control_points);

	// Monte Carlo integration

	std::default_random_engine generator1(1);
	std::uniform_real_distribution<double> distribution1(0.0,1.0);
	const unsigned int rand_number(5000);
	double rainbow_curve_length(0.);
	double rainbow_curve_length_squared(0.);
	for(unsigned int i=0;i<rand_number;++i)
	{
		iga_rt::Point<DIM> jacobian = rainbow_curve.evaluateJacobian(distribution1(generator1));
		const double jacobianSqrMod(jacobian.m_coords[0]*jacobian.m_coords[0]+jacobian.m_coords[1]*jacobian.m_coords[1]);
		rainbow_curve_length += std::sqrt(jacobianSqrMod);
		rainbow_curve_length_squared += jacobianSqrMod;
	}

	rainbow_curve_length /= rand_number;
	const double rainbow_curve_length_error(std::sqrt(((rainbow_curve_length_squared/rand_number)-(rainbow_curve_length*rainbow_curve_length))/rand_number));
	std::cout<<"Numerical integral "<<rainbow_curve_length<<" +- "<<rainbow_curve_length_error<<" / Exact integral "<<0.5*M_PI<<std::endl;

	if(std::abs(rainbow_curve_length-0.5*M_PI)<=3.*rainbow_curve_length_error)
	{
		std::cout<<"Test 1 passed"<<std::endl;
	}
	else
	{
		std::cout<<"Test 1 failed"<<std::endl;
	}


	// Test surface calculation of circle sector

	constexpr double surface_weigths[] = {1.,(1.+1./std::sqrt(2))/2.,(1.+1./std::sqrt(2))/2.,1.,1.,(1.+1./std::sqrt(2))/2.,(1.+1./std::sqrt(2))/2.,1.};

	std::vector<iga_rt::Point<DIM>> rainbow_control_points(8);

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

	iga_rt::NurbsSurface<P1,N1,P2,N2,DIM> rainbow_surface(knots1, knots2, surface_weigths, rainbow_control_points);

	// Monte Carlo integration

	std::default_random_engine generator2(2);
	std::uniform_real_distribution<double> distribution2(0.0,1.0);
	double rainbow_area(0.);
	double rainbow_area_squared(0.);
	for(unsigned int i=0;i<rand_number;++i)
	{
		std::array<iga_rt::Point<DIM>,2> jacobian = rainbow_surface.evaluateJacobian(distribution1(generator1),distribution2(generator2));
		const double jacobianDet(std::abs(jacobian[0].m_coords[0]*jacobian[1].m_coords[1]-jacobian[1].m_coords[0]*jacobian[0].m_coords[1]));
		rainbow_area += jacobianDet;
		rainbow_area_squared += jacobianDet*jacobianDet;
	}

	rainbow_area /= rand_number;
	const double rainbow_area_error(std::sqrt(((rainbow_area_squared/rand_number)-(rainbow_area*rainbow_area))/rand_number));
	std::cout<<"Numerical integral "<<rainbow_area<<" +- "<<rainbow_area_error<<" / Exact integral "<<0.75*M_PI<<std::endl;

	if(std::abs(rainbow_area-0.75*M_PI)<=3.*rainbow_area_error)
	{
		std::cout<<"Test 2 passed"<<std::endl;
	}
	else
	{
		std::cout<<"Test 2 failed"<<std::endl;
	}


	// Test surface calculation of plate with hole

	constexpr double surface_weigthsp[] = {1.,(1.+1./std::sqrt(2))/2.,(1.+1./std::sqrt(2))/2.,1.,1.,1.,1.,1.,1.,1.,1.,1.};
	std::vector<iga_rt::Point<DIM>> plate_control_pointsp(12);

	plate_control_pointsp[0].m_coords[0] = 1.;
	plate_control_pointsp[0].m_coords[1] = 0.;

	plate_control_pointsp[1].m_coords[0] = 1.;
	plate_control_pointsp[1].m_coords[1] = std::sqrt(2.) - 1.;

	plate_control_pointsp[2].m_coords[0] = std::sqrt(2.) - 1.;
	plate_control_pointsp[2].m_coords[1] = 1.;

	plate_control_pointsp[3].m_coords[0] = 0.;
	plate_control_pointsp[3].m_coords[1] = 1.;

	plate_control_pointsp[4].m_coords[0] = 2.5;
	plate_control_pointsp[4].m_coords[1] = 0.;

	plate_control_pointsp[5].m_coords[0] = 2.5;
	plate_control_pointsp[5].m_coords[1] = 0.75;

	plate_control_pointsp[6].m_coords[0] = 0.75;
	plate_control_pointsp[6].m_coords[1] = 2.5;

	plate_control_pointsp[7].m_coords[0] = 0.;
	plate_control_pointsp[7].m_coords[1] = 2.5;

	plate_control_pointsp[8].m_coords[0] = 4.;
	plate_control_pointsp[8].m_coords[1] = 0.;

	plate_control_pointsp[9].m_coords[0] = 4.;
	plate_control_pointsp[9].m_coords[1] = 4.;

	plate_control_pointsp[10].m_coords[0] = 4.;
	plate_control_pointsp[10].m_coords[1] = 4.;

	plate_control_pointsp[11].m_coords[0] = 0.;
	plate_control_pointsp[11].m_coords[1] = 4.;

	iga_rt::NurbsSurface<P1p,N1p,P2,N2,DIM> plate_surface(knots1p, knots2, surface_weigthsp, plate_control_pointsp);

	double plate_area(0.);
	double plate_area_squared(0.);
	for(unsigned int i=0;i<rand_number;++i)
	{
		std::array<iga_rt::Point<DIM>,2> jacobian = plate_surface.evaluateJacobian(distribution1(generator1),distribution2(generator2));
		const double jacobianDet(std::abs(jacobian[0].m_coords[0]*jacobian[1].m_coords[1]-jacobian[1].m_coords[0]*jacobian[0].m_coords[1]));
		plate_area += jacobianDet;
		plate_area_squared += jacobianDet*jacobianDet;
	}

	plate_area /= rand_number;
	const double plate_area_error(std::sqrt(((plate_area_squared/rand_number)-(plate_area*plate_area))/rand_number));
	std::cout<<"Numerical integral "<<plate_area<<" +- "<<plate_area_error<<" / Exact integral "<<16. - 0.25*M_PI<<std::endl;

	if(std::abs(plate_area - 16. + 0.25*M_PI)<=3.*plate_area_error)
	{
		std::cout<<"Test 3 passed"<<std::endl;
	}
	else
	{
		std::cout<<"Test 3 failed"<<std::endl;
	}


	return 0;
}
