#pragma once

#include <tuple>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/adapted/std_tuple.hpp>

// TODO: coord_type required!
struct coord_t
{
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

	double m_x;
	double m_y;
	double m_z;
};

// TODO: overall, can we switch to Variadic templates for better interfaces???
// TODO: servono operatori di somma o unione tra dati geometrici in modo da poter assegnare lo stesso tag all'inisieme di piu' luoghi geometrici

// TODO: derive structure of geometrical data using traits: usando i traits e' possibile stabilire per ogni
// SubdomainDef quanti input servono e di conseguenza quanti parametri prendere dalla lista dei variadic!
// Dovrebbe essere possibile eliminare tutte le strutture intermedie _data

template <typename SubdomainDef>
struct subdomain_geom_data;

// TODO: check if move contructor is used with subdomain object with the following structures!
struct square;
template <>
struct subdomain_geom_data<square>
{

	subdomain_geom_data(coord_t i_up_left_corner,coord_t i_bottom_right_corner):
		m_up_left_corner(i_up_left_corner),
		m_bottom_right_corner(i_bottom_right_corner)
	{}

	inline coord_t up_left_corner(void) const { return m_up_left_corner; }
	inline coord_t& up_left_corner(void) { return m_up_left_corner; }
	inline coord_t bottom_right_corner(void) const { return m_bottom_right_corner; }
	inline coord_t& bottom_right_corner(void) { return m_bottom_right_corner; }

	coord_t m_up_left_corner;
	coord_t m_bottom_right_corner;
};

#if 1

template <typename SubdomainDef>
struct subdomain_data
{
	subdomain_data(double i_tolerance, const subdomain_geom_data<SubdomainDef>& i_geom_data):
		m_tolerance(i_tolerance),
		m_geom_data(i_geom_data)
	{}

	double tolerance(void) const { return m_tolerance; }
	double& tolerance(void) { return m_tolerance; }
	subdomain_geom_data<SubdomainDef> geom_data(void) const { return m_geom_data; }
	subdomain_geom_data<SubdomainDef>& geom_data(void) { return m_geom_data; }

	double m_tolerance;
	subdomain_geom_data<SubdomainDef> m_geom_data;
};

template <typename SubdomainDef>
struct tagged_subdomain_data
{
	// TODO: set as default the copy ctor
	tagged_subdomain_data(const u_int i_tag, const subdomain_data<SubdomainDef>& i_data):
		m_tag(i_tag),
		m_data(i_data)
	{}

	u_int tag(void) const { return m_tag; }
	u_int& tag(void) { return m_tag; }
	subdomain_data<SubdomainDef> data(void) const { return m_data; }
	subdomain_data<SubdomainDef>& data(void) { return m_data; }

	u_int m_tag;
	subdomain_data<SubdomainDef> m_data;
};


template <typename SubdomainDef>
struct subdomain
{
	//TODO: avoid explicit tolerance parameter, it can be stripped from parameter pack and/or using a templated data struct
	//TODO: check on positive tolerance required
	subdomain(subdomain_data<SubdomainDef> const & i_data):
		m_data(i_data)
	{}

	bool is_inside(const coord_t& i_coords) const
	{
		return static_cast<SubdomainDef const &>(*this).is_inside_impl(i_coords);
	}

protected:

	const subdomain_data<SubdomainDef> m_data;
};

// TODO: the idea here is to use these structs to wrap some external geometrical/meshing libraries
// and avoid REAL comp. geometry code implementation...
struct square : public subdomain<square>
{

	//private:// TODO: these should be private

	bool is_inside_impl(const coord_t& i_coords) const
	{
		return ( (i_coords.x()-m_data.geom_data().up_left_corner().x())>m_data.tolerance() &&
				 (m_data.geom_data().bottom_right_corner().x()-i_coords.x())>m_data.tolerance() &&
				 (i_coords.y()-m_data.geom_data().bottom_right_corner().y())>m_data.tolerance() &&
				 (m_data.geom_data().up_left_corner().y()-i_coords.y())>m_data.tolerance() );
	}

};

template<typename SubdomainDef>
struct tagged_subdomain
{
	// TODO: avoid copy of subdomain object
	tagged_subdomain(const tagged_subdomain_data<SubdomainDef>& i_data):
		m_tag(i_data.tag()),
		m_subdomain(i_data.data())
	{}

	tagged_subdomain(const tagged_subdomain& i_tagged_subdomain) = default;

	tagged_subdomain& operator=(const tagged_subdomain& i_tagged_subdomain) = default;

	inline const u_int get_tag(void) const { return m_tag; }

	bool is_inside(const coord_t& i_coords) const { return m_subdomain.is_inside(i_coords); }

	inline subdomain<SubdomainDef> const & get_subdomain(void) const { return m_subdomain; }

private:

	const u_int m_tag;
//	const subdomain<SubdomainDef> m_subdomain; // TODO: cons here!
	subdomain<SubdomainDef> m_subdomain;
};

// TODO: use GT API for this
struct tag_assigner_operator
{
	// TODO: very bad for performance (is move semantics used here?)
	tag_assigner_operator(const coord_t& i_coords, u_int& io_tag):
		m_coords(i_coords),
		m_tag(io_tag)
	{}

	template <typename T>
	void operator()(T const &t) const
	{
		if(!t.get_subdomain().is_inside(m_coords))
		{
			m_tag = t.get_tag();
		}
	}

private:
	const coord_t m_coords;
	u_int& m_tag;
};

// TODO: this should be change into/interfaced to a GT functor, its just a loop over mesh dofs
template <typename FirstSubdomainDef, typename ... OtherSubdomainDefs>
struct bc_tag_assigner
{
	using tagged_subdomains_t = std::tuple<tagged_subdomain<FirstSubdomainDef>, subdomain<OtherSubdomainDefs> ... >;

//	// TODO: avoid tagged subdomain set copy
//	// TODO: use GT API for type tuple generation
	bc_tag_assigner(tagged_subdomain_data<FirstSubdomainDef> const & i_first_subdomain_data,
					tagged_subdomain_data<OtherSubdomainDefs> const & ... i_other_subdomain_data):
		m_tagged_subdomains(std::make_tuple(tagged_subdomain<FirstSubdomainDef>(i_first_subdomain_data),
											tagged_subdomain<OtherSubdomainDefs>(i_other_subdomain_data) ... ))
	{}


	// TODO: only dof coords and tag map is required
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
						// TODO: use GT API loop over tuple elements
						// TODO: use a generic Lambda with auto parameter type (C++14 required)
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

	// TODO: check if data member is required
	tagged_subdomains_t m_tagged_subdomains;
};

struct bc_tag_assigner_static
{
	// TODO: only dof coords and tag map is required
	template <typename Geometry, typename FirstSubdomainDef, typename ... OtherSubdomainDefs>
	void static apply(assembly_base<Geometry>& io_assembly_base,
					  tagged_subdomain_data<FirstSubdomainDef> const & i_first_subdomain_data,
					  tagged_subdomain_data<OtherSubdomainDefs> const & ... i_other_subdomain_data)
	{
		using tagged_subdomains_t = std::tuple<tagged_subdomain<FirstSubdomainDef>, subdomain<OtherSubdomainDefs> ... >;
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
						// TODO: use GT API loop over tuple elements
						// TODO: use a generic Lambda with auto parameter type (C++14 required)
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

struct bc_tag_assigner_gt
{
	// TODO: only dof coords and tag map is required
	template <typename Geometry, typename FirstSubdomainDef, typename ... OtherSubdomainDefs>
	void static apply(assembly_base<Geometry>& io_assembly_base,
					  tagged_subdomain_data<FirstSubdomainDef> const & i_first_subdomain_data,
					  tagged_subdomain_data<OtherSubdomainDefs> const & ... i_other_subdomain_data)
	{
		using tagged_subdomains_t = std::tuple<tagged_subdomain<FirstSubdomainDef>, subdomain<OtherSubdomainDefs> ... >;
		tagged_subdomains_t tagged_subdomains(std::make_tuple(tagged_subdomain<FirstSubdomainDef>(i_first_subdomain_data),
															  tagged_subdomain<OtherSubdomainDefs>(i_other_subdomain_data) ... ));


		// TODO: use GT API loop over tuple elements
		// TODO: use a generic Lambda with auto parameter type (C++14 required)
		u_int current_dof_tag{0};
		tag_assigner_operator_gt tag_assigner(
				coord_t(io_assembly_base.grid()( i, j, k,  dof_index,  0),
						io_assembly_base.grid()( i, j, k,  dof_index,  1),
						io_assembly_base.grid()( i, j, k,  dof_index,  2)),
						current_dof_tag);
		boost::fusion::for_each(tagged_subdomains,tag_assigner);


		for(u_int i=0;i<io_assembly_base.m_d1;++i)
		{
			for(u_int j=0;j<io_assembly_base.m_d2;++j)
			{
				for(u_int k=0;k<io_assembly_base.m_d3;++k)
				{
					for(u_int dof_index=0;dof_index<assembly_base<Geometry>::geo_map::basisCardinality;++dof_index)
					{
						// TODO: use GT API loop over tuple elements
						// TODO: use a generic Lambda with auto parameter type (C++14 required)
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

#else

//// Geometric data traits for size

template <typename SubdomainDef>
struct subdomain_geom_data_traits
{
	static const u_int data_size = 0;
};

template <>
struct subdomain_geom_data_traits<square>
{
	static const u_int data_size = 2;
};

//// Geometric data traits for size

template <typename SubdomainDef>
struct subdomain
{
	//TODO: avoid explicit tolerance parameter, it can be stripped from parameter pack and/or using a templated data struct
	//TODO: check on positive tolerance required
	template <typename ... GeomData>
	subdomain(const double i_tolerance, GeomData const & ... i_geom_data):
		m_tolerance(i_tolerance)
	{
		static_cast<SubdomainDef&>(*this).init_geom_data(i_geom_data ...);
	}

	bool is_inside(const coord_t& i_coords) const
	{
		return static_cast<SubdomainDef const &>(*this).is_inside_impl(i_coords);
	}

protected:

	const double m_tolerance;
};

// TODO: the idea here is to use these structs to wrap some external geometrical/meshing libraries
// and avoid REAL comp. geometry code implementation...
struct square : public subdomain<square>
{

	//private:// TODO: these should be private

	bool is_inside_impl(const coord_t& i_coords) const
	{
		std::cout<<"m_up_left_corner "<<m_up_left_corner.x()<<" "<<m_up_left_corner.y()<<std::endl;
		std::cout<<"m_bottom_right_corner "<<m_bottom_right_corner.x()<<" "<<m_bottom_right_corner.y()<<std::endl;

		return ( (i_coords.x()-m_up_left_corner.x())>m_tolerance &&
				 (m_bottom_right_corner.x()-i_coords.x())>m_tolerance &&
				 (i_coords.y()-m_bottom_right_corner.y())>m_tolerance &&
				 (m_up_left_corner.y()-i_coords.y())>m_tolerance);
	}

	void init_geom_data(coord_t const & i_up_left_corner, coord_t const & i_bottom_right_corner)
	{
		m_up_left_corner = i_up_left_corner;
		m_bottom_right_corner = i_bottom_right_corner;
	}

	// TODO: these should be const
	coord_t m_up_left_corner;
	coord_t m_bottom_right_corner;

};

template<typename SubdomainDef>
struct tagged_subdomain
{
	// TODO: avoid copy of subdomain object
	template <typename ... GeomData>
	tagged_subdomain(const u_int i_tag, const double i_tolerance, GeomData const & ... i_geom_data):
		m_tag(i_tag),
		m_subdomain(i_tolerance,i_geom_data ...)
	{}

	tagged_subdomain(const tagged_subdomain& i_tagged_subdomain) = default;

	tagged_subdomain& operator=(const tagged_subdomain& i_tagged_subdomain) = default;

	inline const u_int get_tag(void) const { return m_tag; }

	bool is_inside(const coord_t& i_coords) const { return m_subdomain.is_inside(i_coords); }

	inline subdomain<SubdomainDef> const & get_subdomain(void) const { return m_subdomain; }

private:

	const u_int m_tag;
//	const subdomain<SubdomainDef> m_subdomain; // TODO: cons here!
	subdomain<SubdomainDef> m_subdomain;
};

// TODO: this should be change into/interfaced to a GT functor, its just a loop over mesh dofs
template <typename FirstSubdomainDef, typename ... OtherSubdomainDefs>
struct bc_tag_assigner
{
	using tagged_subdomains_t = std::tuple<tagged_subdomain<FirstSubdomainDef>, subdomain<OtherSubdomainDefs> ... >;

//	// TODO: avoid tagged subdomain set copy
//	// TODO: use GT API for type tuple generation
	template <typename ... SubdomainData>
	bc_tag_assigner(const u_int i_first_subdomain_tag, const double i_first_subdomain_tolerance, SubdomainData const & ... i_data
			FirstSubdomainData const & i_first_subdomain_data,
					OtherSubdomainData const & ... i_other_subdomain_data):
		m_tagged_subdomains(std::make_tuple(tagged_subdomain<FirstSubdomainDef>(i_first_subdomain_data),
											tagged_subdomain<OtherSubdomainDefs>(i_other_subdomain_data) ... ))
	{}


	// TODO: only dof coords and tag map is required
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
//						// TODO: use GT API loop over tuple elements
//						// TODO: use a generic Lambda with auto parameter type (C++14 required)
//						u_int current_dof_tag{0};
//						tag_assigner_operator tag_assigner(
//								coord_t(io_assembly_base.grid()( i, j, k,  dof_index,  0),
//										io_assembly_base.grid()( i, j, k,  dof_index,  1),
//										io_assembly_base.grid()( i, j, k,  dof_index,  2)),
//										current_dof_tag);
//						boost::fusion::for_each(m_tagged_subdomains,tag_assigner);
//						io_assembly_base.grid_tags()( i, j, k,  dof_index) = current_dof_tag;
					}
				}
			}
		}
	}

	// TODO: check if data member is required
	tagged_subdomains_t m_tagged_subdomains;
};


#endif
