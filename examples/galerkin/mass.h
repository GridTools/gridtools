#pragma once
#include "assembly.h"

using namespace gridtools;

struct integration : public assembly::integration<integration>{

    static auto lhs(uint_t P,, uint_t Q,  uint_t q, uint_t d) -> decltype(p_phi(P,d,q)*p_phi(Q,d,q)) {
        return p_phi(P,d,q)*p_phi(Q,d,q);
    }

    static auto rhs(uint_t P,, uint_t Q,  uint_t q, uint_t d) -> decltype(p_mass(0,0,0,P,Q)) {
        return p_mass(0,0,0,P,Q);
    }
};

int main(){

    assembly::matrix_storage_type mass(d1,d2,d3);
    typedef arg<assembly::size+1, matrix_storage_type> p_mass;
    assembly.domain_append(p_mass(), mass);

    intrepid fe_backend; //! <- setup all the definitions from TPLs
    assembly<intrepid> assembler(fe_backend,d1,d2,d3);

    //! assembles \f$ \int_{\hat\Omega} (J^{-1}\nabla\phi) \cdot (J^{-1}\nabla\psi) |J|\;d{\hat\Omega} \f$
    // from the dimensionality of their storages I can device wether to assemble a matrix or a vector
    // first argument: output
    assembler.template append_esf<integration>(p_mass(), p_dphi(), p_dphi());

    assembler.matrix->ready();
    assembler.matrix->steady();
    assembler.matrix->run();
    assembler.matrix->finalize();
}
