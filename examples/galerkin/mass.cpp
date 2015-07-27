/**
\file
*/
#pragma once
#define PEDANTIC_DISABLED
#include "assembly.h"

    // [integration]
    struct integration {
        using jac_det =accessor<0, range<0,0,0,0> , 4> const;
        using weights =accessor<1, range<0,0,0,0> , 3> const;
        using mass    =accessor<2, range<0,0,0,0> , 5> ;
        using phi    =accessor<3, range<0,0,0,0> , 3> const;
        using psi    =accessor<4, range<0,0,0,0> , 3> const;
        using arg_list= boost::mpl::vector<jac_det, weights, mass, phi, psi> ;
        using quad=Dimension<4>;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {

            quad::Index qp;
            Dimension<5>::Index dimx;
            Dimension<6>::Index dimy;
            // static int_t dd=fe::hypercube_t::boundary_w_codim<2>::n_points::value;

            //projection of f on a (e.g.) P1 FE space ReferenceFESpace1:
            //loop on quadrature nodes, and on nodes of the P1 element (i,j,k) with i,j,k\in {0,1}
            for(short_t P_i=0; P_i<fe::basisCardinality; ++P_i) // current dof
            {
                for(short_t Q_i=0; Q_i<fe::basisCardinality; ++Q_i)
                {//other dofs whose basis function has nonzero support on the element
                    for(short_t q=0; q<cubature::numCubPoints; ++q){
                        eval(mass(0,0,0,P_i,Q_i))  +=
                            eval(!phi(P_i,q,0)*(!psi(Q_i,q,0))*jac_det(qp+q)*!weights(q,0,0));
                    }
                }
            }
        }
    };
    // [integration]

int main(){
    using as=assembly<intrepid::intrepid>;

    auto d1=8;
    auto d2=8;
    auto d3=1;

    intrepid::intrepid fe_backend;
    as assembler(fe_backend,d1,d2,d3);

    as::matrix_type mass_(d1,d2,d3,fe::basisCardinality,fe::basisCardinality);
    typedef arg<as::size+1, as::matrix_type> p_mass;
    domain_type<boost::mpl::push_back<as::accessor_list, p_mass>::type >
        domain(boost::fusion::push_back(assembler.domain().storage_pointers, &mass_));

    mass_.initialize(0.);

    auto computation=make_computation<gridtools::BACKEND, as::layout_t>(
        make_mss
        (
            execute<forward>(),
            make_esf<as::update_jac>( as::p_grid_points(), as::p_jac(), as::p_dphi())
            , make_esf<as::det>(as::p_jac(), as::p_jac_det())
            , make_esf<integration>(as::p_jac_det(), as::p_weights(), p_mass(), as::p_phi(), as::p_phi())
            // , make_esf<as::assembly_f>(p_mass(), p_assembled_stiffness())
            ), domain, assembler.coords());

    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();
}



// #pragma once
// #include "assembly.h"

// using namespace gridtools;


// int main(){

//     intrepid fe_backend; //! <- setup all the definitions from TPLs
//     assembly<intrepid> assembler(fe_backend,d1,d2,d3);

//     assembly<intrepid>::matrix_storage_type mass(d1,d2,d3);
//     typedef arg<assembly::size+1, assembly::matrix_storage_type> p_mass;

//     gridtools::domain_type<boost::mpl::push_back<accessor_list, p_mass> >
//         domain(boost::fusion::push_back(assembler.domain.local_args, &mass));

//     //! assembles \f$ \int_{\hat\Omega} (J^{-1}\nabla\phi) \cdot (J^{-1}\nabla\psi) |J|\;d{\hat\Omega} \f$
//     // from the dimensionality of their storages I can device wether to assemble a matrix or a vector
//     // first argument: output
//     computation=make_computation(
//         assembler::append_esf<mss_t, integration, p_jac_det, p_jac_inv, p_weights, p_mass, p_phi, p_phi>(),
//         domain, assembler.m_coords);


//     computation->ready();
//     computation->steady();
//     computation->run();
//     computation->finalize();
// }
