#pragma once

#include <dolfin.h>
#include <string>

/**
 * @brief Dolfin mesh wrapper class
 */
struct dolfin_mesh
{
    /**
     * @brief constructor (XML dolfin mesh file required)
     */
	explicit dolfin_mesh(const std::string& i_mesh_filename):
			m_mesh(i_mesh_filename)
	{
	}

	inline u_int num_elements(void) const { return m_mesh.num_cells(); }

	inline u_int num_vertices(void) const { return m_mesh.num_vertices(); }

	// TODO: this would require FE traits in real world cases
	inline u_int num_dofs(void) const { return m_mesh.num_vertices(); }

    /**
     * @brief grid initialization method
     * @param io_assembly_base_ I/O parameter, memory allocation must be performed externally
     */
	template <typename Geometry>
	void build_grid(assembly_base<Geometry>& io_assembly_base_)
	{
		const std::vector<unsigned int>& mesh_cells(m_mesh.cells());
		const std::vector<double>& mesh_vert_coords(m_mesh.coordinates());
		const u_short num_cell_vertexes(m_mesh.type().num_vertices());
		const u_short num_space_dim(m_mesh.geometry().dim());

		std::vector<size_t> mesh_cell_orientations(m_mesh.num_cells());
		for(u_int element_index=0;element_index<m_mesh.num_cells();++element_index)
		{
			dolfin::Cell cell(m_mesh, element_index);
			mesh_cell_orientations[element_index] = cell.orientation();
			for(u_short vert_index=0;vert_index<num_cell_vertexes;++vert_index)
			{
				unsigned int global_vert_index(mesh_cells[element_index*num_cell_vertexes + vert_index]);
				// TODO: this would require FE traits in real world cases
				io_assembly_base_.grid_map()( 0,  element_index,  0,  vert_index) = global_vert_index;
				global_vert_index *= num_space_dim;
				for(u_short space_dim=0;space_dim<num_space_dim;++space_dim)
				{
					io_assembly_base_.grid()( 0,  element_index,  0,  vert_index,  space_dim) =
							mesh_vert_coords[global_vert_index + space_dim];
				}
			}
			
			//TODO: can we just check jacobian determinant sign and leave mesh unchanged?
			//TODO: node reordering requires subsequent operations on dolfin mesh to be performed carefully
			//TODO: test with GT functor
			//TODO: implement code also for other elements
			if(mesh_cell_orientations[element_index])
			{
				// Swap among 2nd and 3rd vertex required
				unsigned int global_vertex_index(io_assembly_base_.grid_map()( 0,  element_index,  0,  1));
				io_assembly_base_.grid_map()( 0,  element_index,  0,  1) = io_assembly_base_.grid_map()( 0,  element_index,  0,  2);
				io_assembly_base_.grid_map()( 0,  element_index,  0,  2) = global_vertex_index;

				global_vertex_index = mesh_cells[element_index*num_cell_vertexes + 2]*num_space_dim;
				for(u_short space_dim=0;space_dim<num_space_dim;++space_dim)
				{
					io_assembly_base_.grid()( 0,  element_index,  0,  1,  space_dim) =
							mesh_vert_coords[global_vertex_index + space_dim];
				}

				global_vertex_index = mesh_cells[element_index*num_cell_vertexes + 1]*num_space_dim;
				for(u_short space_dim=0;space_dim<num_space_dim;++space_dim)
				{
					io_assembly_base_.grid()( 0,  element_index,  0,  2,  space_dim) =
							mesh_vert_coords[global_vertex_index + space_dim];
				}
			}
		}
		return;
	}

private:

	dolfin::Mesh m_mesh;

};
