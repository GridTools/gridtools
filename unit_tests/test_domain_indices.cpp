#define BOOST_NO_CXX11_RVALUE_REFERENCES

#include "stencil-composition/make_computation.hpp"
#include <boost/current_function.hpp>

// #ifdef CUDA_EXAMPLE
// #include "stencil-composition/backend_cuda.hpp"
// #else
// #include "stencil-composition/backend_naive.hpp"
// #endif

#include <boost/fusion/include/nview.hpp>
#include <boost/fusion/include/make_vector.hpp>

using namespace gridtools;
using namespace enumtype;

uint_t count;
bool result;

struct print_ {
    print_(void)
    {}

    template <typename T>
    void operator()(T const& v) const {
        if (T::value != count)
            result = false;
        ++count;
    }
};

struct print_plchld {
    mutable uint_t count;
    mutable bool result;

    print_plchld(void)
    {}

    template <typename T>
    void operator()(T const& v) const {
        if (T::index_type::value != count) {
            result = false;
        }
        ++count;
    }
};

bool test_domain_indices() {

    typedef gridtools::backend<enumtype::Host,enumtype::Naive>::storage_type<float_type, storage_info<0,layout_map<0,1,2> > >::type storage_type;
    typedef gridtools::backend<enumtype::Host,enumtype::Naive>::temporary_storage_type<float_type, storage_info<0,layout_map<0,1,2> > >::type tmp_storage_type;


    uint_t d1 = 10;
    uint_t d2 = 10;
    uint_t d3 = 10;

    storage_info<0, layout_map<0,1,2> > meta_(d1,d2,d3);
    storage_type in(meta_,-1., "in");
    storage_type out(meta_,-7.3, "out");
    storage_type coeff(meta_,8., "coeff");

    typedef arg<2, tmp_storage_type > p_lap;
    typedef arg<1, tmp_storage_type > p_flx;
    typedef arg<5, tmp_storage_type > p_fly;
    typedef arg<0, storage_type > p_coeff;
    typedef arg<3, storage_type > p_in;
    typedef arg<4, storage_type > p_out;

    result = true;

    typedef boost::mpl::vector<p_lap, p_flx, p_fly, p_coeff, p_in, p_out> accessor_list;

    domain_type<accessor_list> domain
       (boost::fusion::make_vector(&out, &in, &coeff /*,&fly, &flx*/));

    count = 0;
    result = true;

    print_plchld pfph;
    count = 0;
    result = true;
    boost::mpl::for_each<domain_type<accessor_list>::placeholders>(pfph);


    return result;
}
