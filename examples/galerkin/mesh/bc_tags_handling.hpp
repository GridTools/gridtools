#pragma once

#include <tuple>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/adapted/std_tuple.hpp>

// TODO: common coord_t object required!
// TODO: variadic templates for better interfaces: instead of domain_data like structures we could use parameter pack,
// their size for each subdomain shape can be described using traits
// TODO: set operators (union, intersection, etc) required for flexible description of boundary regions
// TODO: move ctor/assignement operator when data structures are being used

// [coord_t]
/**
   @class point coordinates in 3D space
 */
struct coord_t
{
	double m_x;
	double m_y;
	double m_z;

	coord_t(const double i_x, const double i_y, const double i_z = 0.0):
		m_x(i_x),
		m_y(i_y),
		m_z(i_z)
	{}

	inline double& x(void) { return m_x; }
	inline double& y(void) { return m_y; }
	inline double& z(void) { return m_z; }

	inline double x(void) const { return m_x; }
	inline double y(void) const { return m_y; }
	inline double z(void) const { return m_z; }

	inline double mod2(void) const { return m_x*m_x + m_y*m_y + m_z*m_z; }

	inline double mod(void) const { return std::sqrt(m_x*m_x + m_y*m_y + m_z*m_z); }

};

coord_t operator-(const coord_t& i_point1, const coord_t& i_point2)
{
	return coord_t(i_point1.x() - i_point2.x(),
				   i_point1.y() - i_point2.y(),
				   i_point1.z() - i_point2.z());
}

double operator*(const coord_t& i_point1, const coord_t& i_point2)
{
	return i_point1.x()*i_point2.x() + i_point1.y()*i_point2.y() + i_point1.z()*i_point2.z();
}

coord_t operator^(const coord_t& i_point1, const coord_t& i_point2)
{
	return coord_t(i_point1.y()*i_point2.z() - i_point1.z()*i_point2.y(),
				   i_point1.z()*i_point2.x() - i_point1.x()*i_point2.z(),
				   i_point1.x()*i_point2.y() - i_point1.y()*i_point2.x());
}

inline double mod2(const coord_t& i_coord) { return i_coord.mod2(); }

inline double mod(const coord_t& i_coord) { return i_coord.mod(); }
// [coord_t]


// [subdomain_geom_data]
struct segment;
struct square;

/**
   @struct Geometry data for subdomain definition
   @tparam SubdomainDef subdomain definition
 */
template <typename SubdomainDef>
struct subdomain_geom_data;

//TODO: factorize common structs (e.g. in square and segment)
template <>
struct subdomain_geom_data<square>
{

	const coord_t m_up_left_corner;
	const coord_t m_bottom_right_corner;

	subdomain_geom_data(coord_t i_up_left_corner,coord_t i_bottom_right_corner):
		m_up_left_corner(i_up_left_corner),
		m_bottom_right_corner(i_bottom_right_corner)
	{}

	inline coord_t up_left_corner(void) const { return m_up_left_corner; }
	inline coord_t bottom_right_corner(void) const { return m_bottom_right_corner; }

};

template <>
struct subdomain_geom_data<segment>
{
	const coord_t m_start_point;
	const coord_t m_stop_point;
	const coord_t m_segment_vec;
	const double m_segment_len;

	subdomain_geom_data(coord_t i_start_point,coord_t i_stop_point):
		m_start_point(i_start_point),
		m_stop_point(i_stop_point),
		m_segment_vec(i_stop_point - i_start_point),
		m_segment_len(m_segment_vec.mod())
	{}

	inline coord_t start_point(void) const { return m_start_point; }
	inline coord_t stop_point(void) const { return m_stop_point; }
	inline coord_t segment_vec(void) const { return m_segment_vec; }
	inline double segment_len(void) const { return m_segment_len; }

};
// [subdomain_geom_data]

// [subdomain_data]
/**
   @struct Geometry + tolerance data for subdomain definition
   @tparam SubdomainDef subdomain definition
 */
template <typename SubdomainDef>
struct subdomain_data
{
	double m_tolerance;
	subdomain_geom_data<SubdomainDef> m_geom_data;

	//TODO: check on positive tolerance required
	subdomain_data(double i_tolerance, const subdomain_geom_data<SubdomainDef>& i_geom_data):
		m_tolerance(i_tolerance),
		m_geom_data(i_geom_data)
	{}

	double tolerance(void) const { return m_tolerance; }
	double& tolerance(void) { return m_tolerance; }
	subdomain_geom_data<SubdomainDef> geom_data(void) const { return m_geom_data; }
	subdomain_geom_data<SubdomainDef>& geom_data(void) { return m_geom_data; }

};
// [subdomain_data]

// [tagged_subdomain_data]
/**
   @struct Geometry + tolerance + tag data for subdomain definition
   @tparam SubdomainDef subdomain definition
 */
template <typename SubdomainDef>
struct tagged_subdomain_data
{
	u_int m_tag;
	subdomain_data<SubdomainDef> m_data;

	tagged_subdomain_data(const u_int i_tag, const subdomain_data<SubdomainDef>& i_data):
		m_tag(i_tag),
		m_data(i_data)
	{}

	u_int tag(void) const { return m_tag; }
	u_int& tag(void) { return m_tag; }
	subdomain_data<SubdomainDef> data(void) const { return m_data; }
	subdomain_data<SubdomainDef>& data(void) { return m_data; }
};
// [tagged_subdomain_data]

// [subdomain]
/**
   @struct subdomain geometry definition struct
   @tparam SubdomainDef subdomain definition
 */
template <typename SubdomainDef>
struct subdomain
{

	const subdomain_data<SubdomainDef> m_data;

	subdomain(subdomain_data<SubdomainDef> const & i_data):
		m_data(i_data)
	{}

	/**
	 * @brief check if point is in subdomain boundary method
	 * @param i_coords point coordinates
	 * @return true if point is on subdomain boundary (according to @tparam SubdomainDef definition of boundary)
	 */
	bool is_on_boundary(const coord_t& i_coords) const
	{
		return static_cast<SubdomainDef const &>(*this).is_on_boundary_impl(i_coords);
	}

};
// [subdomain]


// TODO: the idea here is to use these structs to wrap some external geometrical/meshing libraries
// and avoid REAL comp. geometry code implementation...
/**
   @struct square subdomain (parallel to x-y axis and on z=0 plan)
 */
struct square : public subdomain<square>
{

	//private:// TODO: these should be private

	bool is_on_boundary_impl(const coord_t& i_coords) const
	{
		if(is_inside_impl(i_coords))
		{
			return false;
		}

		if((i_coords.x()-m_data.geom_data().up_left_corner().x())>-m_data.tolerance() &&
		   (m_data.geom_data().bottom_right_corner().x()-i_coords.x())>-m_data.tolerance())
		{
			if(std::abs((i_coords.y()-m_data.geom_data().up_left_corner().y()))<m_data.tolerance() ||
			   std::abs((i_coords.y()-m_data.geom_data().bottom_right_corner().y()))<m_data.tolerance())
			{
				return true;
			}
		}

		if((i_coords.y()-m_data.geom_data().bottom_right_corner().y())>-m_data.tolerance() &&
		   (m_data.geom_data().up_left_corner().y()-i_coords.y())>-m_data.tolerance())
		{
			if(std::abs((i_coords.x()-m_data.geom_data().up_left_corner().x()))<m_data.tolerance() ||
			   std::abs((i_coords.x()-m_data.geom_data().bottom_right_corner().x()))<m_data.tolerance())
			{
				return true;
			}
		}

		return false;
	}

	bool is_inside_impl(const coord_t& i_coords) const
	{
		return ( (i_coords.x()-m_data.geom_data().up_left_corner().x())>m_data.tolerance() &&
				 (m_data.geom_data().bottom_right_corner().x()-i_coords.x())>m_data.tolerance() &&
				 (i_coords.y()-m_data.geom_data().bottom_right_corner().y())>m_data.tolerance() &&
				 (m_data.geom_data().up_left_corner().y()-i_coords.y())>m_data.tolerance() );
	}

};

/**
   @struct segment in 3D subdomain
 */
struct segment : public subdomain<segment>
{

	//private:// TODO: these should be private

	bool is_on_boundary_impl(const coord_t& i_coords) const
	{
		const coord_t start_to_point_vec(coord_t(i_coords - m_data.geom_data().start_point()));
		const double cross_prod_mod2(mod2(start_to_point_vec^m_data.geom_data().segment_vec()));

		if(cross_prod_mod2>m_data.tolerance()*m_data.tolerance())
		{
			return false;
		}

		const double scal_prod(start_to_point_vec*m_data.geom_data().segment_vec());

		if(scal_prod<-m_data.tolerance())
		{
			return false;
		}

		if(start_to_point_vec.mod()>m_data.geom_data().segment_len()+m_data.tolerance())
		{
			return false;
		}

		return true;
	}

};
// [subdomain]


// [tagged_subdomain]
/**
   @struct subdomain with tag definition struct
   @tparam SubdomainDef subdomain definition
 */
template<typename SubdomainDef>
struct tagged_subdomain
{
	const u_int m_tag;
//	const subdomain<SubdomainDef> m_subdomain; // TODO: cons here!
	subdomain<SubdomainDef> m_subdomain;

	tagged_subdomain(const tagged_subdomain_data<SubdomainDef>& i_data):
		m_tag(i_data.tag()),
		m_subdomain(i_data.data())
	{}

	tagged_subdomain(const tagged_subdomain& i_tagged_subdomain) = default;

	tagged_subdomain& operator=(const tagged_subdomain& i_tagged_subdomain) = default;

	inline const u_int get_tag(void) const { return m_tag; }

	/**
	 * @brief check if boundary tag must be assigned to point method
	 * @param i_coords point coordinates
	 * @return true if point is on subdomain boundary (according to @tparam SubdomainDef definition of boundary)
	 */
	bool assign_tag(const coord_t& i_coords) const { return m_subdomain.is_on_boundary(i_coords); }

};
// [tagged_subdomain]

// [tag_assigner_operator]
// TODO: use GT API for this
/**
   @struct operator() definition struct for tag assignement operator for boost_for_each
 */
struct tag_assigner_operator
{
	const coord_t m_coords;
	u_int& m_tag;


	tag_assigner_operator(const coord_t& i_coords, u_int& io_tag):
		m_coords(i_coords),
		m_tag(io_tag)
	{}

	template <typename T>
	void operator()(T const &t) const
	{
		if(t.assign_tag(m_coords))
		{
			m_tag = t.get_tag();
		}
	}

};
// [tag_assigner_operator]

// [bc_tag_assigner]
// TODO: this should be change into/interfaced to a GT functor, its just a loop over mesh dofs
/**
   @struct mesh dof tag boundary condition tag assigner struct
   @tparam FirstSubdomainDef first subdomain definition
   @tparam OtherSubdomainDefs remaining subdomain definitions
 */
template <typename FirstSubdomainDef, typename ... OtherSubdomainDefs>
struct bc_tag_assigner
{
	using tagged_subdomains_t = std::tuple<tagged_subdomain<FirstSubdomainDef>, subdomain<OtherSubdomainDefs> ... >;

	tagged_subdomains_t m_tagged_subdomains;

//	// TODO: use GT API for type tuple generation
	bc_tag_assigner(tagged_subdomain_data<FirstSubdomainDef> const & i_first_subdomain_data,
					tagged_subdomain_data<OtherSubdomainDefs> const & ... i_other_subdomain_data):
		m_tagged_subdomains(std::make_tuple(tagged_subdomain<FirstSubdomainDef>(i_first_subdomain_data),
											tagged_subdomain<OtherSubdomainDefs>(i_other_subdomain_data) ... ))
	{
		// TODO: add static asserts (see wiki)
	}


	// TODO: only dof coords and tag map is required here in input
	/**
	   @brief tag assignement execution method
	   @tparam Geometry
	   @param dof grid
	 */
	template <typename Geometry>
	void apply(assembly_base<Geometry>& io_assembly_base)
	{
		for(u_int i=0;i<io_assembly_base.m_d1;++i)
		{
			for(u_int j=0;j<io_assembly_base.m_d2;++j)
			{
				for(u_int k=0;k<io_assembly_base.m_d3;++k)
				{
					for(u_int dof_index=0;dof_index<assembly_base<Geometry>::geo_map::basisCardinality;++dof_index)
					{
						// TODO: use GT API for loop over tuple elements
						// TODO: use a generic Lambda with auto parameter type (C++14 required)
						// TODO: fix "current_dof_tag"
						u_int current_dof_tag{0};
						tag_assigner_operator tag_assigner(
								coord_t(io_assembly_base.grid()( i, j, k,  dof_index,  0),
										io_assembly_base.grid()( i, j, k,  dof_index,  1),
										io_assembly_base.grid()( i, j, k,  dof_index,  2)),
										current_dof_tag);
						boost::fusion::for_each(m_tagged_subdomains,tag_assigner);
						io_assembly_base.grid_tags()( i, j, k,  dof_index) = current_dof_tag;
					}
				}
			}
		}
	}

};

// TODO: this should be change into/interfaced to a GT functor, its just a loop over mesh dofs
/**
   @struct mesh dof tag boundary condition tag assigner struct
   @tparam FirstSubdomainDef first subdomain definition
   @tparam OtherSubdomainDefs remaining subdomain definitions
 */
struct bc_tag_assigner_static
{
	// TODO: only dof coords and tag map is required
	// TODO: only dof coords and tag map is required here in input from grid data
	/**
	   @brief tag assignement execution method
	   @tparam Geometry
	   @tparam FirstSubdomainDef first subdomain definition
	   @tparam OtherSubdomainDefs remaining subdomain definitions
	   @param dof grid
	   @param first subdomain definition data
	   @param remaining subdomain definition data
	 */
	template <typename Geometry, typename FirstSubdomainDef, typename ... OtherSubdomainDefs>
	void static apply(assembly_base<Geometry>& io_assembly_base,
					  tagged_subdomain_data<FirstSubdomainDef> const & i_first_subdomain_data,
					  tagged_subdomain_data<OtherSubdomainDefs> const & ... i_other_subdomain_data)
	{
		// TODO: add static asserts (see wiki)

		using tagged_subdomains_t = std::tuple<tagged_subdomain<FirstSubdomainDef>, tagged_subdomain<OtherSubdomainDefs> ... >;
		tagged_subdomains_t tagged_subdomains(std::make_tuple(tagged_subdomain<FirstSubdomainDef>(i_first_subdomain_data),
															  tagged_subdomain<OtherSubdomainDefs>(i_other_subdomain_data) ... ));

		for(u_int i=0;i<io_assembly_base.m_d1;++i)
		{
			for(u_int j=0;j<io_assembly_base.m_d2;++j)
			{
				for(u_int k=0;k<io_assembly_base.m_d3;++k)
				{
					for(u_int dof_index=0;dof_index<assembly_base<Geometry>::geo_map::basisCardinality;++dof_index)
					{
						// TODO: use GT API for loop over tuple elements
						// TODO: use a generic Lambda with auto parameter type (C++14 required)
						// TODO: fix "current_dof_tag"
						u_int current_dof_tag{0};
						tag_assigner_operator tag_assigner(
								coord_t(io_assembly_base.grid()( i, j, k,  dof_index,  0),
										io_assembly_base.grid()( i, j, k,  dof_index,  1),
										io_assembly_base.grid()( i, j, k,  dof_index,  2)),
										current_dof_tag);
						boost::fusion::for_each(tagged_subdomains,tag_assigner);
						io_assembly_base.grid_tags()( i, j, k,  dof_index) = current_dof_tag;
					}
				}
			}
		}
	}
};
// [bc_tag_assigner]

