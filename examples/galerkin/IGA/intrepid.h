#pragma once

#include <gridtools.h>

#include "shards_topology"

#include <Intrepid_Basis.hpp>
#include <Intrepid_Types.hpp>
#include <Intrepid_Cubature.hpp>
#include <Intrepid_FieldContainer.hpp>
#include <Intrepid_DefaultCubatureFactory.hpp>

//just some of the possible discretizations
#include "Intrepid_HGRAD_TET_Cn_FEM_ORTH.hpp"
#include "Intrepid_HGRAD_TET_Cn_FEM.hpp"
#include "Intrepid_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"
#include "Intrepid_HGRAD_TET_C1_FEM.hpp"
#include "Intrepid_HGRAD_TRI_C1_FEM.hpp"
#include "Intrepid_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid_HGRAD_TET_C2_FEM.hpp"
#include "Intrepid_HGRAD_TRI_C2_FEM.hpp"
#include "Intrepid_HGRAD_HEX_C2_FEM.hpp"
#include "Intrepid_HGRAD_QUAD_C2_FEM.hpp"
#include "Intrepid_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"

/** @file
    @brief in this file is implemented a minimal interface for using Intrepid (Trilinos) discretizations in GridTools */

//! This policy has the only goal of uniforming the construction of the basis in Intrepid. In fact the constructors for the basis
//! can accept zero, 1 or 2 compile-time ocnstants as arguments. Using a policy allows to select the interface baed on the type of basis chosen.
namespace gridtools{

    template<class BasisType, uint_t order=1>
    class basis_type_policy : public BasisType
    {
    public:
        typedef BasisType basis_type;
        basis_type_policy():
            basis_type()
            {}
    };

//!
    template<class Scalar, class ArrayScalar, uint_t order>
    class basis_type_policy<Intrepid::Basis_HGRAD_TET_Cn_FEM_ORTH<Scalar, ArrayScalar>, order> : public Intrepid::Basis_HGRAD_TET_Cn_FEM_ORTH<Scalar, ArrayScalar>
    {
    public:
        typedef  Intrepid::Basis_HGRAD_TET_Cn_FEM_ORTH<Scalar, ArrayScalar> basis_type;
        //template<uint_t Order>
        basis_type_policy():
            basis_type(order)
            {}
        //    instantiate(){return basis_type(/*Order*/5);}
    };


    template<class Scalar, class ArrayScalar, uint_t order>
    class basis_type_policy<Intrepid::Basis_HGRAD_TET_Cn_FEM<Scalar, ArrayScalar>, order> : public Intrepid::Basis_HGRAD_TET_Cn_FEM<Scalar, ArrayScalar>
    {
    public:
        typedef  Intrepid::Basis_HGRAD_TET_Cn_FEM<Scalar, ArrayScalar> basis_type;
        //template<uint_t Order>
        basis_type_policy():
            basis_type(order,Intrepid::POINTTYPE_EQUISPACED)
            {}
        //    instantiate(){return basis_type(/*Order*/5);}
    };


    template<class Scalar, class ArrayScalar, uint_t order>
    class basis_type_policy<Intrepid::Basis_HGRAD_TRI_Cn_FEM<Scalar, ArrayScalar>, order> : public Intrepid::Basis_HGRAD_TRI_Cn_FEM<Scalar, ArrayScalar>
    {
    public:
        typedef  Intrepid::Basis_HGRAD_TRI_Cn_FEM<Scalar, ArrayScalar> basis_type;
        //template<uint_t Order>
        basis_type_policy():
            basis_type(order,Intrepid::POINTTYPE_EQUISPACED)
            {}
        //    instantiate(){return basis_type(/*Order*/5);}
    };


    template<class Scalar, class ArrayScalar, uint_t order>
    class basis_type_policy<Intrepid::Basis_HGRAD_LINE_Cn_FEM<Scalar, ArrayScalar>, order> : public Intrepid::Basis_HGRAD_LINE_Cn_FEM<Scalar, ArrayScalar>
    {
    public:
        typedef  Intrepid::Basis_HGRAD_LINE_Cn_FEM<Scalar, ArrayScalar> basis_type;
        //template<uint_t Order>
        basis_type_policy():
            basis_type(order,Intrepid::POINTTYPE_EQUISPACED)
            {}
        //    instantiate(){return basis_type(/*Order*/5);}
    };



    template<class ArrayScalar, class CellTopologyType, class BasisType , int quadratureDegree>
    class intrepid
    {
        typedef ArrayScalar array_scalar_t;
    private:

        /*!ArrayScalar is a tempate argument of Intrepid_Basis. It usually points to the Intrepid::FieldContainer,
          which is a container that allows to store a small matrix internally as a vector
          (pretty much like the old SimpleVector in LifeV), it allows several algebraic operations and has the possiblility
          of enabling a lot of error checking for indexes out of bound*/
        //! No way to use the copy constuctor
        intrepid( intrepid const& );

        array_scalar_t m_phi;
        array_scalar_t m_dphi;
        array_scalar_t m_d2phi;
        array_scalar_t m_div_phi;
        uint_t m_nb_quad_pts ;
        std::shared_ptr<basis_type> m_basis_ptr;
        Intrepid::DefaultCubatureFactory<value_t> m_quadrature_factory;
        Teuchos::RCP<Intrepid::Cubature<value_t> > m_quadrature;
        Intrepid::FieldContainer<value_t> m_quad_pts;
        Intrepid::FieldContainer<value_t> m_quad_weights;
        const uint_t m_nb_dofs;

    public:
        typedef CellTopologyType cell_topology_t;
        typedef BasisType basis_t;
        typedef typename BasisType::Scalar value_t ;
        //typedef to typename BasisType::Scalar not allowed by the C++ standard
        typedef  Intrepid::FieldContainer<value_t> array_scalar_t ;

        //! @name Constructor & Destructor
        //@{

        //! Empty constructor
        /*!
         */
        intrepid():
            m_phi(),
            m_dphi(),
            m_d2phi(),
            m_div_phi(),
            m_nb_quad_pts(0),
            m_basis_ptr(new basis_type_policy<BasisType, order>()),
            m_quadrature_factory(),
            m_quadrature(m_quadratureFactory.create(cell_type, quadratureDegree)),
            m_quad_pts(),
            m_quad_weights(),
            m_nb_dofs(m_basisPtr->getCardinality())
            {
                m_quadrature->getCubature(m_quad_points, m_quad_weights);
            }

        //! Destructor
        virtual ~intrepid(){}

        //@}

        //   //! @name Methods
        //   //@{
        static
        uint_t nb_quad_pts( )
            {
                return m_nb_quad;
            }

        //@}

        //! @name Get Methods
        //@{

        //! Return the name of the reference element.
        static const std::string& name(){
            return basis_type::EBasisToString(basis_type::basisType_);
        }

        //! Return the number of degrees of freedom for this reference element
        const uint_t& nb_dofs(){
            return m_nb_dof;
        }

        //! Return the number of local coordinates
        const uint_t& nb_coords(){
            return cell_topology_t::s_dimensions;
        }

        //! Return the quadrature
        const Intrepid::Cubature<value_t>&  quadrature(){
            return *m_quadrature;
        }

        //! Return the cubature weights
        const Intrepid::FieldContainer<value_t>&  quadrature_weights(){
            return m_quad_weights;
        }

        scalar_array_t& phi(){return m_phi;}
        scalar_array_t& dphi(){return m_dphi;}
        scalar_array_t& d2phi(){return m_d2phi;}
        scalar_array_t& div_phi(){return m_div_phi;}
        //@}


        //! These are the operations that can be computed in the reference element
        /*!
          This method can be called from the main file: if we have to compute a mass matrix we'd need to initialize the OPERATOR_VALUE, while
          if we have to assemble a stiffness matrix we'd need the first derivatives, OPERATOR_D1, and so on. Other elemental operators
          (e.g. OPERATOR_CURL) are available in Intrepid, check them out.
        */
        bool initialize_variables(Intrepid::EOperator oper){
            switch (oper)
            {
            case Intrepid::OPERATOR_VALUE:
                m_phi=array_scalar_t(m_nb_dofs, m_nb_quad_pts, 1);
                m_basis_ptr->getValues( m_phi, m_quad_points, Intrepid::OPERATOR_VALUE);
                return true;
            case Intrepid::OPERATOR_D1:
                m_dphi=array_scalar_t(m_nb_dofs, m_nb_quad_pts, nb_coords());
                m_basis_ptr->getValues( m_dphi, m_quad_points, Intrepid::OPERATOR_D1);
                return true;
            case Intrepid::OPERATOR_D2:
                m_d2phi=array_scalar_t(m_nb_dofs, m_nb_quad_pts, nb_coords()*2);
                /*In Intrepid the mixed derivatives are stored in the last 3 coordinates of a vector of dimension 6 (that's why nbcoor()*2)*/
                m_basis_ptr->getValues( m_d2phi, m_quad_points, Intrepid::OPERATOR_D2);
                return true;
            case Intrepid::OPERATOR_DIV:
                m_div_phi=array_scalar_t(m_nb_dofs, m_nb_quad_pts, 1);
                m_basis_ptr->getValues( m_div_phi, S_quadPoints, Intrepid::OPERATOR_DIV);
                return true;
            default :
                return false;
            }
        }

        //! returns the quadrature weight given the quadrature nood
        static
        value_t quadrature_weight( uint_t coor ){
            return m_quad_weights(coor);
        }

        //! returns the cell topology.
        constexpr
        auto shape() -> const decltype(*cell_topology_t::s_topology) & {return *cell_topology_t::s_topology;}

        //@}
    };

    // template<class AS, class CT, class BT , int QD>
    // const uint_t intrepid<AS, CT, BT, QD>::S_nbDof=;

    template <typename T>
    struct is_reference_space ;

    template <class CellTopologyType, class BasisType , ushort_t quadratureDegree, ushort_t order>
    struct is_reference_space<intrepid<CellTopologyType, BasisType , quadratureDegree, order> > : boost::mpl::true{};
}//gridtools
