#include <cassert>
#include "../functors/matvec.hpp"
#include "../galerkin_defs.hpp"
// TODO: we need an specific header for storage type definition to avoid the include that follows
#include "../numerics/basis_functions.hpp"
#include "stencil-composition/stencil-composition.hpp"
#include "stencil-composition/axis.hpp"

//using namespace gridtools;
//using namespace gdl;
//using namespace gdl::enumtype;

typedef gridtools::interval<gridtools::level<0,-2>, gridtools::level<1,1> > axis;

int main() {

    //![problem_size]
    constexpr unsigned int d1=5;
    constexpr unsigned int d2=5;
    constexpr unsigned int d3=5;
    constexpr unsigned int n_rows(10);
    //![problem_size]

    //![storages]
    // vector a
    using vector_a_storage_info_t=gdl::storage_info< __COUNTER__, gdl::layout_tt<4> >;
    using vector_a_type=gdl::storage_t< vector_a_storage_info_t >;
    vector_a_storage_info_t v_a_(d1,d2,d3,n_rows);
    vector_a_type v_a(v_a_, 0.e0, "v_a");
    // vector b
    using vector_b_storage_info_t=gdl::storage_info< __COUNTER__, gdl::layout_tt<4> >;
    using vector_b_type=gdl::storage_t< vector_b_storage_info_t >;
    vector_b_storage_info_t v_b_(d1,d2,d3,n_rows);
    vector_b_type v_b(v_b_, 0.e0, "v_b");
    // vector out
    using vector_out_storage_info_t=gdl::storage_info< __COUNTER__, gdl::layout_tt<4> >;
    using vector_out_type=gdl::storage_t< vector_out_storage_info_t >;
    vector_out_storage_info_t v_out_(d1,d2,d3,n_rows);
    vector_out_type v_out(v_out_, 0.e0, "v_out");
    //![storages]

    //![storage_initializazion]
    for (gdl::uint_t i=0; i<d1; i++)
        for (gdl::uint_t j=0; j<d2; j++)
            for (gdl::uint_t k=0; k<d3; k++)
                for (gdl::uint_t row=0; row<n_rows; ++row) {
                    v_a(i,j,k,row) = 1.0;
                    v_b(i,j,k,row) = -1.0;
                    v_out(i,j,k,row) = 0.0;
                }
    //![storage_initializazion]


    //![grid]
    auto grid=gridtools::grid<axis>({1, 0, 1, d1-1, d1},
                                         {1, 0, 1, d2-1, d2});
    grid.value_list[0] = 1;
    grid.value_list[1] = d3-1;
    //![grid]


    //![domain]
    typedef gdl::gt::arg<0,vector_a_type> p_v_a;
    typedef gdl::gt::arg<1,vector_b_type> p_v_b;
    typedef gdl::gt::arg<2,vector_out_type> p_v_out;
    typedef boost::mpl::vector<p_v_a, p_v_b, p_v_out> accessor_list;
    ::gridtools::aggregator_type<accessor_list> domain(boost::fusion::make_vector(&v_a, &v_b, &v_out));
    //![domain]

    //![computation]
    auto compute=::gridtools::make_computation<BACKEND>(domain,
                                                      grid,
                                                      gridtools::make_multistage(gridtools::enumtype::execute<gridtools::enumtype::forward>(),
                                                                                 gridtools::make_stage< gdl::functors::vecvec<4,gdl::functors::sum_operator<float_t> > >(p_v_a(),p_v_b(),p_v_out())
                                                                                 )
                                                     );
    compute->ready();
    compute->steady();
    compute->run();
    compute->finalize();
    //![computation]


    //![check_results]
    bool success = true;
    for (gdl::uint_t i=0; i<d1; i++)
        for (gdl::uint_t j=0; j<d2; j++)
            for (gdl::uint_t k=0; k<d3; k++)
                for (gdl::uint_t row=0; row<n_rows; ++row)
                    if(v_a(i,j,k,row) + v_b(i,j,k,row) != v_out(i,j,k,row)) {
                        success = false;
                        break;
                    }
    //![check_results]


    assert(success);

}
