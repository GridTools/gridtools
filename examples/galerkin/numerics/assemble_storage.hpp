#pragma once

// TODO: I am including this header because it contains storage_t definition, why not using separate headers for galerking related defs?
#include "basis_functions.hpp"
#include "../utils/indexing.hpp"
#include <iostream>

namespace gdl {

    /**
      @class DOF/Mesh element adjacency rule application struct

          Given a pair of mesh element coords and dof in element coords for a given
          direction, we need to know if the corresponding global dof pair belong to
          a single element (because at least one of the do is located on an adjacency
          region) or they are "disconnected". In this second case the corresponding
          assemble matrix value is zero since the related basis function supports
          do not overlap. This condition depends on the specific mesh element shape.

      @tparam Grid specific rule implementation struct (CRTP)
     */
    template <typename GridSpecificRuleApplier>
    struct grid_adjacency_rule_applier
    {

        /**
          @brief adjacency rule application method

            This method performs the check described in struct comment and, if possible,
            provides updated values for the element/dof coords, corresponding to the case
            of a dof pair that can be seen as a single element case, assuming that coord values
            in the other direction are the same. In this case the true value is returned.

          @param First mesh element coordinate (in a given direction, the same of the other input pars)
          @param First dof coordinate in mesh element
          @param Second mesh element coordinate
          @param Second dof coordinate in mesh element
          @return true if the dof pair belong to a single mesh element, false otherwise
         */
        static inline bool apply(ushort_t& io_X1, ushort_t& io_x1, ushort_t& io_X2, ushort_t& io_x2)
        {
            return GridSpecificRuleApplier::apply_impl(io_X1,io_x1,io_X2,io_x2);
        }

        // TODO: unify
        static inline bool apply(ushort_t& io_X1, ushort_t& io_x1)
        {
            return GridSpecificRuleApplier::apply_impl(io_X1,io_x1);
        }


    };

    /**
      @class DOF/Mesh element adjacency rule application implementation case for hexahedron mesh
     */
    template<ushort_t BasisCardinality1D>
    struct hexa_grid_adjacency_rule_applier : public grid_adjacency_rule_applier< hexa_grid_adjacency_rule_applier<BasisCardinality1D> >
    {
        /**
          @brief adjacency rule application method implementation (see grid_adjacency_rule_applier comment)

          @param First mesh element coordinate (in a given direction, the same of the other input pars) (I/O par)
          @param First dof coordinate in mesh element (I/O par)
          @param Second mesh element coordinate (I/O par)
          @param Second dof coordinate in mesh element (I/O par)
          @return true if the dof pair belong to a single mesh element, false otherwise (I/O par)
         */
        static bool apply_impl(ushort_t& io_X1, ushort_t& io_x1, ushort_t& io_X2, ushort_t& io_x2)
        {
            // TODO: fix all uns short comparison operations
            if((io_X1 >1 + io_X2) || (io_X2 >1 + io_X1))
            {
                return false;
            }
            else if(io_X2  == io_X1 + 1)
            {
                if(io_x2 == 0)
                {
                    io_X2--;
                    io_x2 = BasisCardinality1D-1;
                }
                else
                {
                    return false;
                }
            }
            else if(io_X1 == io_X2 + 1 )
            {
                if(io_x1 == 0)
                {
                    io_X1--;
                    io_x1 = BasisCardinality1D-1;
                }
                else
                {
                    return false;
                }
            }

            return true;
        }

    };

    /**
      @class Halo data storing structure

          This structure stores the data concerning halo
          structure in storage object allowing the correct
          access of "real" data location and skipping the
          halo ones

     */
    struct halo_data {

        /**
          @brief constructor
          @param i_halo_size_x_full Total number of halo element in x (first) direction
          @param i_halo_size_y_full Total number of halo element in y (second) direction
          @param i_halo_size_z_full Total number of halo element in z (third) direction
          @param i_halo_size_x_left Number of "left" (i.e., near the origin) halo element in x (first) direction
          @param i_halo_size_y_left Number of "left" (i.e., near the origin) halo element in y (second) direction
          @param i_halo_size_z_left Number of "left" (i.e., near the origin) halo element in z (third) direction
         */
        halo_data(uint_t i_halo_size_x_full = 0,
                  uint_t i_halo_size_y_full = 0,
                  uint_t i_halo_size_z_full = 0,
                  uint_t i_halo_size_x_left = 0,
                  uint_t i_halo_size_y_left = 0,
                  uint_t i_halo_size_z_left = 0):
                  m_halo_size_x_full(i_halo_size_x_full),
                  m_halo_size_y_full(i_halo_size_y_full),
                  m_halo_size_z_full(i_halo_size_z_full),
                  m_halo_size_x_left(i_halo_size_x_left),
                  m_halo_size_y_left(i_halo_size_y_left),
                  m_halo_size_z_left(i_halo_size_z_left)
        {}

        // TODO: avoid usge of x,y,z?
        const uint_t m_halo_size_x_full;
        const uint_t m_halo_size_y_full;
        const uint_t m_halo_size_z_full;
        const uint_t m_halo_size_x_left;
        const uint_t m_halo_size_y_left;
        const uint_t m_halo_size_z_left;
    };


    /**
      @class Global to local dof translation struct

          This struct apply method performs a traslation between of global dof coordinates
          providing the corresponding (grid element coords, local dof coords) position pair
          that can be used to access assemble_storage data.

      @tparam basis cardinality in x direction
      @tparam basis cardinality in y direction
      @tparam basis cardinality in z direction
     */
    template <ushort_t BasisCardinality0, ushort_t BasisCardinality1 = BasisCardinality0, ushort_t BasisCardinality2 = BasisCardinality0>
    struct global_to_local_dof_translator {

    private:

        static constexpr ushort_t s_max_dof_index0{BasisCardinality0-1};
        static constexpr ushort_t s_max_dof_index1{BasisCardinality1-1};
        static constexpr ushort_t s_max_dof_index2{BasisCardinality2-1};

        const uint_t m_grid_dim0;
        const uint_t m_grid_dim1;
        const uint_t m_grid_dim2;

        const uint_t m_total_dof0;
        const uint_t m_total_dof1;
        const uint_t m_total_dof2;

        const halo_data m_halo_data;

    public:

        /**
          @brief constructor
          @param i_grid_dim0 number of mesh elements in x direction
          @param i_grid_dim1 number of mesh elements in y direction
          @param i_grid_dim2 number of mesh elements in z direction
          @param i_grid_dim2 number of mesh elements in z direction
          @param i_halo_data storage halo data
         */
        global_to_local_dof_translator(const uint_t i_grid_dim0, const uint_t i_grid_dim1, const uint_t i_grid_dim2, const halo_data& i_halo_data):
            m_grid_dim0(i_grid_dim0 - i_halo_data.m_halo_size_x_full),
            m_grid_dim1(i_grid_dim1 - i_halo_data.m_halo_size_y_full),
            m_grid_dim2(i_grid_dim2 - i_halo_data.m_halo_size_z_full),
            // A +1 should be present here but then we would need (m_total_dof0 - 1) in the code below
            m_total_dof0(s_max_dof_index0*m_grid_dim0),
            m_total_dof1(s_max_dof_index1*m_grid_dim1),
            m_total_dof2(s_max_dof_index2*m_grid_dim2),
            m_halo_data(i_halo_data)
            {}

        /**
          @brief translation method

              This method performs the traslation described in struct comment.
              Additionally, it returns a translation status concerning dof positioning
              according to dof owner elements as discussed in grid_adjacency_rule_applier
              struct description. When the status is false the corresponding assemble
              data element is considered equal to zero.

          @param first dof global coord in x direction
          @param first dof global coord in y direction
          @param first dof global coord in z direction
          @param second dof global coord in x direction
          @param second dof global coord in y direction
          @param second dof global coord in z direction
          @param first dof owner element coord in x direction (Output par)
          @param first dof owner element coord in y direction (Output par)
          @param first dof owner element coord in z direction (Output par)
          @param first dof local coord in x direction
          @param first dof local coord in y direction
          @param first dof local coord in z direction
          @param second dof owner element coord in x direction (Output par)
          @param second dof owner element coord in y direction (Output par)
          @param second dof owner element coord in z direction (Output par)
          @param second dof local coord in x direction
          @param second dof local coord in y direction
          @param second dof local coord in z direction
          @return true if dof pair exist on a single mesh element, false otherwise
         */
        // TODO: second dof owner is never used, remove it from function signature
        bool apply(ushort_t i_Id1, ushort_t i_Jd1, ushort_t i_Kd1, ushort_t i_Id2, ushort_t i_Jd2, ushort_t i_Kd2,
                   ushort_t& io_I1, ushort_t& io_J1, ushort_t& io_K1, ushort_t& io_i1, ushort_t& io_j1, ushort_t& io_k1,
                   ushort_t& io_I2, ushort_t& io_J2, ushort_t& io_K2, ushort_t& io_i2, ushort_t& io_j2, ushort_t& io_k2) const
        {
            if(i_Id1<m_total_dof0)
            {
                io_I1 = i_Id1/s_max_dof_index0 + m_halo_data.m_halo_size_x_left;
                io_i1 = i_Id1%s_max_dof_index0;
            }
            else
            {
                io_I1 = m_grid_dim0 - 1 + m_halo_data.m_halo_size_x_left;
                io_i1 = s_max_dof_index0;
            }

            if(i_Jd1<m_total_dof1)
            {
                io_J1 = i_Jd1/s_max_dof_index1 + m_halo_data.m_halo_size_y_left;
                io_j1 = i_Jd1%s_max_dof_index1;
            }
            else
            {
                io_J1 = m_grid_dim1 - 1 + m_halo_data.m_halo_size_y_left;
                io_j1 = s_max_dof_index1;
            }

            if(i_Kd1<m_total_dof2)
            {
                io_K1 = i_Kd1/s_max_dof_index2 + m_halo_data.m_halo_size_z_left;
                io_k1 = i_Kd1%s_max_dof_index2;
            }
            else
            {
                io_K1 = m_grid_dim2 - 1 + m_halo_data.m_halo_size_z_left;
                io_k1 = s_max_dof_index2;
            }

            if(i_Id2<m_total_dof0)
            {
                io_I2 = i_Id2/s_max_dof_index0 + m_halo_data.m_halo_size_x_left;
                io_i2 = i_Id2%s_max_dof_index0;
            }
            else
            {
                io_I2 = m_grid_dim0 - 1 + m_halo_data.m_halo_size_x_left;
                io_i2 = s_max_dof_index0;
            }

            if(i_Jd2<m_total_dof1)
            {
                io_J2 = i_Jd2/s_max_dof_index1 + m_halo_data.m_halo_size_y_left;
                io_j2 = i_Jd2%s_max_dof_index1;
            }
            else
            {
                io_J2 = m_grid_dim1 - 1 + m_halo_data.m_halo_size_y_left;
                io_j2 = s_max_dof_index1;
            }

            if(i_Kd2<m_total_dof2)
            {
                io_K2 = i_Kd2/s_max_dof_index2 + m_halo_data.m_halo_size_z_left;
                io_k2 = i_Kd2%s_max_dof_index2;
            }
            else
            {
                io_K2 = m_grid_dim2 - 1 + m_halo_data.m_halo_size_z_left;
                io_k2 = s_max_dof_index2;
            }


            if(grid_adjacency_rule_applier< hexa_grid_adjacency_rule_applier<BasisCardinality0> >::apply(io_I1,io_i1,io_I2,io_i2) == false)
            {
                return false;
            }

            if(grid_adjacency_rule_applier< hexa_grid_adjacency_rule_applier<BasisCardinality1> >::apply(io_J1,io_j1,io_J2,io_j2) == false)
            {
                return false;
            }

            if(grid_adjacency_rule_applier< hexa_grid_adjacency_rule_applier<BasisCardinality2> >::apply(io_K1,io_k1,io_K2,io_k2) == false)
            {
                return false;
            }

            return true;
        }

        // TODO: add comment and rename variables
        bool apply(ushort_t i_Id1, ushort_t i_Jd1, ushort_t i_Kd1, ushort_t& io_I1, ushort_t& io_J1, ushort_t& io_K1, ushort_t& io_i1, ushort_t& io_j1, ushort_t& io_k1) const
        {
            if(i_Id1<m_total_dof0)
            {
                io_I1 = i_Id1/s_max_dof_index0 + m_halo_data.m_halo_size_x_left;
                io_i1 = i_Id1%s_max_dof_index0;
            }
            else
            {
                io_I1 = m_grid_dim0 - 1 + m_halo_data.m_halo_size_x_left;
                io_i1 = s_max_dof_index0;
            }

            if(i_Jd1<m_total_dof1)
            {
                io_J1 = i_Jd1/s_max_dof_index1 + m_halo_data.m_halo_size_y_left;
                io_j1 = i_Jd1%s_max_dof_index1;
            }
            else
            {
                io_J1 = m_grid_dim1 - 1 + m_halo_data.m_halo_size_y_left;
                io_j1 = s_max_dof_index1;
            }

            if(i_Kd1<m_total_dof2)
            {
                io_K1 = i_Kd1/s_max_dof_index2 + m_halo_data.m_halo_size_z_left;
                io_k1 = i_Kd1%s_max_dof_index2;
            }
            else
            {
                io_K1 = m_grid_dim2 - 1 + m_halo_data.m_halo_size_z_left;
                io_k1 = s_max_dof_index2;
            }

            return true;
        }

    };


    /**
      @class FEM assemble focused storage

          This struct extends the GT storage object in order to provide a DOF based
          access to assemble matrix elements calculated making use of the in place
          gathering calculation strategy. Access to the base storage is provided so
          that this storage can be used for GT functions

      @tparam GT storage info
      @tparam basis cardinality in x direction
      @tparam basis cardinality in y direction
      @tparam basis cardinality in z direction
     */
    // TODO: add template parameter for grid traits (we need different access strategy for different 2D/3D gride types)
    // TODO: update parameter names according to GT rules
    template <typename MetaData, ushort_t BasisCardinality0, ushort_t BasisCardinality1 = BasisCardinality0, ushort_t BasisCardinality2 = BasisCardinality0>
    struct assemble_storage : public storage_t< MetaData >::basic_type {

        GRIDTOOLS_STATIC_ASSERT((BasisCardinality0==BasisCardinality1), "GDL Error: basis cardinality in first and second direction must be the same");
        GRIDTOOLS_STATIC_ASSERT((BasisCardinality0==BasisCardinality2), "GDL Error: basis cardinality in first and second direction must be the same");

        using storage_metadata = MetaData;
        using base_storage_type = typename storage_t< MetaData >::basic_type;
        using indexing_t = gt::meta_storage_base<static_int<__COUNTER__>,gt::layout_map<2,1,0>,false>;

    private:

        const global_to_local_dof_translator<BasisCardinality0,BasisCardinality1,BasisCardinality2> m_local_dof;
        // TODO: this should be a constrexpr
        const indexing_t m_indexing;

        // TODO: why s_?
        const uint_t s_max_dof_2;
        const uint_t s_max_dof_1;

        const halo_data m_halo_data;

    public:

        /**
          @brief constructor

              Input and template parameters correspond to those expected by storage_t constructor
         */
        // TODO: should i_halo_data be included in storage_info or variadic data?
        template <typename ... ExtraArgs>
        assemble_storage(const MetaData* i_storage_info, const halo_data& i_halo_data, ExtraArgs const &... i_args):
                         base_storage_type(i_storage_info, i_args ...),
                         m_local_dof(i_storage_info->template dim<0>(),
                                     i_storage_info->template dim<1>(),
                                     i_storage_info->template dim<2>(),
                                     i_halo_data),
                         m_halo_data(i_halo_data),
                         m_indexing(BasisCardinality0,BasisCardinality1,BasisCardinality2),
                         s_max_dof_1((BasisCardinality0-1)*(i_storage_info->template dim<0>()-i_halo_data.m_halo_size_x_full)),
                         s_max_dof_2((BasisCardinality1-1)*(i_storage_info->template dim<1>()-i_halo_data.m_halo_size_y_full))
                         {}


        /**
          @brief global dof base data access method
          @param first dof global coord in x direction
          @param first dof global coord in y direction
          @param first dof global coord in z direction
          @param second dof global coord in x direction
          @param second dof global coord in y direction
          @param second dof global coord in z direction
          @return assemble matrix value for the provided dof coords
         */
        // TODO: more generic interface needed (2D grids will have 4 indexes)
        // TODO: other access tipe should be possible: (dof1,dof2) for example
        // TODO: add input check
        // TODO: this should be const
        GT_FUNCTION
        float_type get_value(ushort_t i_Id1, ushort_t i_Jd1, ushort_t i_Kd1, ushort_t i_Id2, ushort_t i_Jd2, ushort_t i_Kd2) //const
        {
            ushort_t I1;
            ushort_t J1;
            ushort_t K1;
            ushort_t i1;
            ushort_t j1;
            ushort_t k1;
            ushort_t I2;
            ushort_t J2;
            ushort_t K2;
            ushort_t i2;
            ushort_t j2;
            ushort_t k2;

            // TODO: check the following if and return
            if(m_local_dof.apply(i_Id1,i_Jd1,i_Kd1,i_Id2,i_Jd2,i_Kd2,I1,J1,K1,i1,j1,k1,I2,J2,K2,i2,j2,k2))
                return base_storage_type::operator()(I1,J1,K1,m_indexing.index(i1,j1,k1),m_indexing.index(i2,j2,k2));

            return 0;
        }

        float_type& set_value(ushort_t i_Id1, ushort_t i_Jd1, ushort_t i_Kd1, ushort_t i_Id2, ushort_t i_Jd2, ushort_t i_Kd2) //const
        {
            ushort_t I1;
            ushort_t J1;
            ushort_t K1;
            ushort_t i1;
            ushort_t j1;
            ushort_t k1;
            ushort_t I2;
            ushort_t J2;
            ushort_t K2;
            ushort_t i2;
            ushort_t j2;
            ushort_t k2;

//            // TODO: check the following if and return
//            if(m_local_dof.apply(i_Id1,i_Jd1,i_Kd1,i_Id2,i_Jd2,i_Kd2,I1,J1,K1,i1,j1,k1,I2,J2,K2,i2,j2,k2))
            m_local_dof.apply(i_Id1,i_Jd1,i_Kd1,i_Id2,i_Jd2,i_Kd2,I1,J1,K1,i1,j1,k1,I2,J2,K2,i2,j2,k2);
            return base_storage_type::operator()(I1,J1,K1,m_indexing.index(i1,j1,k1),m_indexing.index(i2,j2,k2));

//            return 0;
        }


        // TODO: unify with case above and add dimensional checks
        float_type get_value(ushort_t i_Id1, ushort_t i_Jd1, ushort_t i_Kd1) //const
        {
            ushort_t I1;
            ushort_t J1;
            ushort_t K1;
            ushort_t i1;
            ushort_t j1;
            ushort_t k1;

//            std::cout<<"get value "<<i_Id1<<" "<<i_Jd1<<" "<<i_Kd1<<std::endl;

            if(m_local_dof.apply(i_Id1,i_Jd1,i_Kd1,I1,J1,K1,i1,j1,k1))
                return base_storage_type::operator()(I1,J1,K1,m_indexing.index(i1,j1,k1));

            return 0;
        }

        // TODO: unify with case above and add dimensional checks
        float_type& set_value(ushort_t i_Id1, ushort_t i_Jd1, ushort_t i_Kd1) //const
        {
            ushort_t I1;
            ushort_t J1;
            ushort_t K1;
            ushort_t i1;
            ushort_t j1;
            ushort_t k1;

            //            std::cout<<"get value "<<i_Id1<<" "<<i_Jd1<<" "<<i_Kd1<<std::endl;

            m_local_dof.apply(i_Id1,i_Jd1,i_Kd1,I1,J1,K1,i1,j1,k1);
            return base_storage_type::operator()(I1,J1,K1,m_indexing.index(i1,j1,k1));

        }

        /**
          @brief global dof base data access method
          @param first dof
          @param second dof
          @return assemble matrix value for the provided global dof pair
         */
        // TODO: add input check
        // TODO: this should be const
        GT_FUNCTION
        float_type get_value(uint_t i_I, uint_t i_J) //const
        {

            const ushort_t k1 = i_I/((s_max_dof_1+1)*(s_max_dof_2+1));
            const ushort_t k2 = i_J/((s_max_dof_1+1)*(s_max_dof_2+1));

            const ushort_t j1 = (i_I%((s_max_dof_1+1)*(s_max_dof_2+1)))/(s_max_dof_1+1);
            const ushort_t j2 = (i_J%((s_max_dof_1+1)*(s_max_dof_2+1)))/(s_max_dof_1+1);

            const ushort_t i1 = (i_I%((s_max_dof_1+1)*(s_max_dof_2+1)))%(s_max_dof_1+1);
            const ushort_t i2 = (i_J%((s_max_dof_1+1)*(s_max_dof_2+1)))%(s_max_dof_1+1);

            return get_value(i1, j1, k1, i2, j2, k2);
        }

        float_type& set_value(uint_t i_I, uint_t i_J) //const
        {

            const ushort_t k1 = i_I/((s_max_dof_1+1)*(s_max_dof_2+1));
            const ushort_t k2 = i_J/((s_max_dof_1+1)*(s_max_dof_2+1));

            const ushort_t j1 = (i_I%((s_max_dof_1+1)*(s_max_dof_2+1)))/(s_max_dof_1+1);
            const ushort_t j2 = (i_J%((s_max_dof_1+1)*(s_max_dof_2+1)))/(s_max_dof_1+1);

            const ushort_t i1 = (i_I%((s_max_dof_1+1)*(s_max_dof_2+1)))%(s_max_dof_1+1);
            const ushort_t i2 = (i_J%((s_max_dof_1+1)*(s_max_dof_2+1)))%(s_max_dof_1+1);

            return set_value(i1, j1, k1, i2, j2, k2);
        }

        // TODO: unify with case above
        float_type get_value(uint_t i_I) //const
        {

            const ushort_t k1 = i_I/((s_max_dof_1+1)*(s_max_dof_2+1));

            const ushort_t j1 = (i_I%((s_max_dof_1+1)*(s_max_dof_2+1)))/(s_max_dof_1+1);

            const ushort_t i1 = (i_I%((s_max_dof_1+1)*(s_max_dof_2+1)))%(s_max_dof_1+1);

            return get_value(i1, j1, k1);
        }

        // TODO: unify with case above
        float_type& set_value(uint_t i_I) //const
        {

            const ushort_t k1 = i_I/((s_max_dof_1+1)*(s_max_dof_2+1));

            const ushort_t j1 = (i_I%((s_max_dof_1+1)*(s_max_dof_2+1)))/(s_max_dof_1+1);

            const ushort_t i1 = (i_I%((s_max_dof_1+1)*(s_max_dof_2+1)))%(s_max_dof_1+1);

            return set_value(i1, j1, k1);
        }


    };

}

namespace gridtools{
    template <typename MetaData, ushort_t ... Cardinalities>
    struct is_any_iterate_domain_storage < ::gdl::assemble_storage<MetaData, Cardinalities ...> >
        : boost::mpl::true_ {};

}
