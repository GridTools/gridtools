/**
\file testing the gathering of the information from the neighbours
*/
#define PEDANTIC_DISABLED
#define HAVE_INTREPID_DEBUG
#include <stencil-composition/stencil-composition.hpp>
#include "../numerics/intrepid.hpp"
#include "../numerics/assembly.hpp"
#include "../functors/assembly_functors.hpp"
#include "gather_reference.hpp"
#include <tools/verifier.hpp>

int main(){
    //![definitions]
    using namespace enumtype;
    using namespace gridtools;
    //defining the assembler, based on the Intrepid definitions for the numerics
    using matrix_storage_info_t=storage_info< __COUNTER__, layout_tt<4> >;// TODO: or 3?
    using matrix_type=storage_t< matrix_storage_info_t >;

    using geo_map=reference_element<1, Lagrange, Hexa>;
    using cub=cubature<geo_map::order, geo_map::shape>;
    using geo_t = intrepid::geometry<geo_map, cub>;

    //![boundary]

    //![definitions]

    //dimensions of the problem (in number of elements per dimension)
    auto d1=3;
    auto d2=3;
    auto d3=3;

    //![instantiation_stiffness]
    //defining the stiffness matrix: d1xd2xd3 elements
    matrix_storage_info_t meta_in_(d1,d2,d3,geo_map::basisCardinality);
    //    matrix_storage_info_t meta_out_(d1,d2,d3,geo_t::bd_geo_map::basisCardinality*4);
    matrix_storage_info_t meta_out_(d1,d2,d3,geo_map::basisCardinality);
    matrix_type in_(meta_in_, 1., "in");
    matrix_type out_(meta_out_, 0., "out");
    for (int i=0; i<d1; ++i)
        for (int j=0; j<d2; ++j)
            for (int k=1; k<d3; ++k)
            {
                for (int basis_=0; basis_<geo_map::basisCardinality; ++basis_)
                {
		  in_(i,j,k,basis_) = i+j*10+k*100
                    out_(i,j,k,basis_) = i+j*10+k*100;
                }
            }
    //![placeholders]
    // defining the placeholder for the mass
    typedef arg<0, matrix_type> p_in;
    typedef arg<1, matrix_type> p_out;

    typedef boost::mpl::vector<p_in, p_out> list_t;
    // appending the placeholders to the list of placeholders already in place
    auto domain=aggregator_type< list_t >(boost::fusion::make_vector(&in_, &out_) );
    //![placeholders]


    auto coords=grid<axis>({1, 0, 1, (uint_t)d1-1, (uint_t)d1},
        {1, 0, 1, (uint_t)d2-1, (uint_t)d2});
    coords.value_list[0] = 1;
    coords.value_list[1] = d3-1;

    //![computation]
    auto computation=make_computation< gridtools::BACKEND >(
        make_multistage
        (
            execute<forward>()
            , make_stage< functors::assemble< geo_t > >(p_in(), p_in(), p_out())
            ), domain, coords);

    computation->ready();
    computation->steady();
    computation->run();
    computation->finalize();

    // verify with the reference
    matrix_type reference_(meta_out_, 0., "out");
    reference_1(reference_, in_, d1, d2, d3);
    int retval=0;

    for (int i=1; i<d1; ++i)
        for (int j=1; j<d2; ++j)
            for (int k=1; k<d3; ++k)
            {
                for (int basis_=0; basis_<geo_map::basisCardinality; ++basis_)
                {
                    if(reference_(i,j,k,basis_) != out_(i,j,k,basis_))
                    {
                        std::cout<<out_(i,j,k,basis_)<<" != "<<reference_(i,j,k,basis_)<<" in ("<<i<<" "<<j<<" "<<k<<" "<<basis_<<"); " ;
                        std::cout<<std::endl;
                        retval=-1;
                    }
                }
            }


    return retval;
}
