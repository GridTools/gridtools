#include "b_splines_surface_rt.h"
#include "nurbs_surface_rt.h"
#include "point.h"
#include <vector>
#include <array>
#include <iostream>

int main()
{
	// TODO: remove full function print out and add test checks

	const int P1(1);
	const int N1(2);
	const int P2(2);
	const int N2(4);
	const int DIM(2);
	const std::array<double,P1+N1+1> knots1{0.,0.,1.,1.};
	const std::array<double,P2+N2+1> knots2{0.,0.,0.,0.5,1.,1.,1.};

	std::vector<iga_rt::Point<DIM> > rainbow_control_points(8);

	rainbow_control_points[0].m_coords[0] = 1.;
	rainbow_control_points[0].m_coords[1] = 0.;

	rainbow_control_points[1].m_coords[0] = 1.;
	rainbow_control_points[1].m_coords[1] = 0.5;

	rainbow_control_points[2].m_coords[0] = 0.5;
	rainbow_control_points[2].m_coords[1] = 1.;

	rainbow_control_points[3].m_coords[0] = 0.;
	rainbow_control_points[3].m_coords[1] = 1.;

	rainbow_control_points[4].m_coords[0] = 2.;
	rainbow_control_points[4].m_coords[1] = 0.;

	rainbow_control_points[5].m_coords[0] = 2.;
	rainbow_control_points[5].m_coords[1] = 1.;

	rainbow_control_points[6].m_coords[0] = 1.;
	rainbow_control_points[6].m_coords[1] = 2.;

	rainbow_control_points[7].m_coords[0] = 0.;
	rainbow_control_points[7].m_coords[1] = 2.;

#if 1
	iga_rt::BSplineSurface<P1,N1,P2,N2,DIM> rainbow_surface(knots1, knots2, rainbow_control_points);
#else
	const double weights[] = {0.1,1.,1.,1.,0.1,1.,1.,1.};
	iga_rt::NurbsSurface<P1,N1,P2,N2,DIM> rainbow_surface(knots1, knots2, weights, rainbow_control_points);
#endif

	double minCsi(0.);
	double maxCsi(1.);
	double minEta(0.);
	double maxEta(1.);
	int numPoints(100);
	double deltaCsi =(maxCsi-minCsi)/numPoints;
	double deltaEta =(maxEta-minEta)/numPoints;
	for(double csi=minCsi;csi<=maxCsi;csi+=deltaCsi)
	{
		for(double eta=minEta;eta<=maxEta;eta+=deltaEta)
		{
			const iga_rt::Point<DIM> point(rainbow_surface.evaluate(csi,eta));
			std::cout<<csi<<" "<<eta<<" "<<point.m_coords[0]<<" "<<point.m_coords[1]<<" "<<csi*csi<<std::endl;
		}
	}


	return 0;
}
