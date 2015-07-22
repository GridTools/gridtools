#pragma once
#include "assembly.h"

using namespace gridtools;
struct integration : public assembly::integration{

    template<typename Evaluation>
    Do(Evaluation eval){
        eval(p_mass)=eval(apply(p_phi*p_phi));
    }
};

int main(){

    assembly::matrix_storage_type mass(d1,d2,d3);
    typedef arg<assembly::size+1, matrix_storage_type> p_mass;
    assembly.domain_append(p_mass(), mass)
    assembly assembler(d1,d2,d3);
    assembler.matrix->append_esf(
        make_esf<integration>(p_phi(), p_phi(), p_mass())
        );

    assembler.matrix->ready();
    assembler.matrix->steady();
    assembler.matrix->run();
    assembler.matrix->finalize();
}
