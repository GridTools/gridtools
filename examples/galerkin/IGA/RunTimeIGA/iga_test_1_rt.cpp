#include "b_splines_basis_rt.h"
#include <iostream>
#include <cmath>
#include <array>
#include <numeric>

//TODO: add test description
//TODO: switch to google testing suite

int main()
{


	const int P(2);
	const int N(4);
	const std::array<double,P+N+1> knots{0.,0.,0.,0.5,1.,1.,1.};

	const double csi_test_1 = 0.25;
	const double csi_test_2 = 0.75;
	const double csi_test_3 = 1.;
	const unsigned int knot_index_test_1 = 3-1;
	const unsigned int knot_index_test_2 = 4-1;
	const unsigned int knot_index_test_3 = 4-1;

	// UNIVARIATE CASE

	iga_rt::BSplineBasis<P,N> bsplineTest(knots);

	// Run span finding test 1
	if(bsplineTest.find_span(csi_test_1) == knot_index_test_1)
	{
		std::cout<<"Test 1 passed"<<std::endl;
	}
	else
	{
		std::cout<<"Test 1 failed"<<std::endl;
	}

	// Run span finding test 2
	if(bsplineTest.find_span(csi_test_2) == knot_index_test_2)
	{
		std::cout<<"Test 2 passed"<<std::endl;
	}
	else
	{
		std::cout<<"Test 2 failed"<<std::endl;
	}

	// Run span finding test 1
	if(bsplineTest.find_span(csi_test_3) == knot_index_test_3)
	{
		std::cout<<"Test 3 passed"<<std::endl;
	}
	else
	{
		std::cout<<"Test 3 failed"<<std::endl;
	}

	// Unity test loop
	bool test_passed(true);
	const double unity_tolerance(1.e-6);
	const double minCsi(0.);
	const double maxCsi(1.);
	const int numPoints(1000);
	const double deltaCsi =(maxCsi-minCsi)/numPoints;
	for(double csi=minCsi;csi<=maxCsi;csi+=deltaCsi)
	{
		std::vector<double> b_spline_values(bsplineTest.evaluate(csi));
		if(std::abs(std::accumulate(b_spline_values.begin(),b_spline_values.end(),0.) - 1.)>unity_tolerance)
		{
			test_passed = false;
			break;
		}
	}
	if(test_passed == true)
	{
		std::cout<<"Unity 1D test passed"<<std::endl;
	}
	else
	{
		std::cout<<"Unity 1D test failed"<<std::endl;
	}

	// BIVARIATE CASE

	iga_rt::BivariateBSplineBasis<P,N,P,N> bivariateBsplineTest(knots,knots);

	// Run span finding test 1
	if((bivariateBsplineTest.find_span(csi_test_1,csi_test_1)[0] == knot_index_test_1) &&
	   (bivariateBsplineTest.find_span(csi_test_1,csi_test_1)[1] == knot_index_test_1))
	{
		std::cout<<"Test 4 passed"<<std::endl;
	}
	else
	{
		std::cout<<"Test 4 failed"<<std::endl;
	}

	// Run span finding test 2
	if((bivariateBsplineTest.find_span(csi_test_2,csi_test_2)[0] == knot_index_test_2) &&
	   (bivariateBsplineTest.find_span(csi_test_2,csi_test_2)[1] == knot_index_test_2))
	{
		std::cout<<"Test 5 passed"<<std::endl;
	}
	else
	{
		std::cout<<"Test 5 failed"<<std::endl;
	}

	// Run span finding test 1
	if((bivariateBsplineTest.find_span(csi_test_3,csi_test_3)[0] == knot_index_test_3) &&
	   (bivariateBsplineTest.find_span(csi_test_3,csi_test_3)[1] == knot_index_test_3))
	{
		std::cout<<"Test 6 passed"<<std::endl;
	}
	else
	{
		std::cout<<"Test 6 failed"<<std::endl;
	}

	// Unity test loop
	test_passed = true;
	const double minEta(0.);
	const double maxEta(1.);
	const double deltaEta =(maxEta-minEta)/numPoints;
	for(double csi=minCsi;csi<=maxCsi;csi+=deltaCsi)
	{
		for(double eta=minEta;eta<=maxEta;eta+=deltaEta)
		{
			std::vector<double> b_spline_values(bivariateBsplineTest.evaluate(csi,eta));
			if(std::abs(std::accumulate(b_spline_values.begin(),b_spline_values.end(),0.) - 1.)>unity_tolerance)
			{
				test_passed = false;
				break;
			}
		}
	}
	if(test_passed == true)
	{
		std::cout<<"Unity 2D test passed"<<std::endl;
	}
	else
	{
		std::cout<<"Unity 2D test failed"<<std::endl;
	}

	return 0;
}
