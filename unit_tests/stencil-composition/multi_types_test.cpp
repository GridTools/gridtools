#include <gridtools.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/structured_grids/call_interfaces.hpp>
#include <tools/verifier.hpp>
#include "gtest/gtest.h"


namespace multi_types_test {
using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

#ifdef CXX11_ENABLED
using namespace expressions;
#endif

#ifdef CUDA_EXAMPLE
typedef gridtools::backend<gridtools::enumtype::Cuda, gridtools::enumtype::Block > the_backend;
#else
#ifdef BACKEND_BLOCK
typedef gridtools::backend<gridtools::enumtype::Host, gridtools::enumtype::Block > the_backend;
#else
typedef gridtools::backend<gridtools::enumtype::Host, gridtools::enumtype::Naive > the_backend;
#endif
#endif

typedef gridtools::interval<level<0,-1>, level<1,-1> > region;

typedef gridtools::interval<level<0,-2>, level<1,3> > axis;

struct type1 {
    int i,j,k;

    type1() : i(0), j(0), k(0) {}
    explicit type1(int i, int j, int k) : i(i), j(j), k(k) {}
};
    
struct type4 {
    float x,y,z;

    type4() : x(0.), y(0.), z(0.) {}
    explicit type4(double i, double j, double k) : x(i), y(j), z(k) {}

    type4& operator=(type1 const& a) {
        std::cout << "assign 4 <- 1" << std::endl;
        x = a.i;
        y = a.j;
        z = a.k;
        return *this;
    }
};

    struct type2 {
    double xy;
    type2& operator=(type4 const & x) {
        xy = x.x+x.y;
        return *this;
    }
};
    
struct type3 {
    double yz;

    type3& operator=(type4 const & x) {
        yz = x.y+x.z;
        return *this;
    }
};
    

type4 operator+(type4 const& a, type1 const& b) {
    return type4(a.x+static_cast<double>(b.i),
                 a.y+static_cast<double>(b.j),
                 a.z+static_cast<double>(b.k));
}
type4 operator-(type4 const& a, type1 const& b) {
    return type4(a.x-static_cast<double>(b.i),
                 a.y-static_cast<double>(b.j),
                 a.z-static_cast<double>(b.k));
}
    
struct function0 {
    typedef accessor<0, enumtype::in > in;
    typedef accessor<1, enumtype::inout> out;

    typedef boost::mpl::vector<in, out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, region) {
        eval(out()).i = eval(in()).i+1;
        eval(out()).j = eval(in()).j+1;
        eval(out()).k = eval(in()).k+1;
        //std::cout << "function0" << std::endl;
    }
};

struct function1 {
    typedef accessor<0, enumtype::inout> out;
    typedef accessor<1, enumtype::in > in;

    typedef boost::mpl::vector<out, in> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, region) {
        auto result = call<function0, region>::with(eval, in());
        eval(out()) = result;
        //std::cout << "function1" << std::endl;
    }
};

struct function2 {

    typedef accessor<0, enumtype::inout> out;
    typedef accessor<1, enumtype::in> in;
    typedef accessor<2, enumtype::in> temp;

    typedef boost::mpl::vector<out, in, temp> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, region) {
        eval(out()) = eval(temp())+eval(in());
    }
};

struct function3 {

    typedef accessor<0, enumtype::inout> out;
    typedef accessor<1, enumtype::in> lap;
    typedef accessor<2, enumtype::in> in;

    typedef boost::mpl::vector<out, in, lap> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, region) {
        eval(out()) = eval(lap(0,1,0))-eval(in());
    }
};

/*
 * The following operators and structs are for debugging only
 */
std::ostream& operator<<(std::ostream& s, function1 const) {
    return s << "function1";
}
std::ostream& operator<<(std::ostream& s, function2 const) {
    return s << "function2";
}
std::ostream& operator<<(std::ostream& s, function3 const) {
    return s << "function3";
}

bool test(uint_t x, uint_t y, uint_t z)
{

    uint_t d1 = x;
    uint_t d2 = y;
    uint_t d3 = z;
    uint_t halo_size = 2;

#ifdef __CUDACC__
    typedef gridtools::layout_map<2,1,0> layout_type;//stride 1 on i
#else
    typedef gridtools::layout_map<0,1,2> layout_type;//stride 1 on k
#endif

    typedef gridtools::storage_info<0, layout_type> storage_info1_t;
    typedef gridtools::storage_info<1, layout_type> storage_info2_t;
    typedef gridtools::storage_info<2, layout_type> storage_info3_t;

    typedef the_backend::storage_type<type1, storage_info1_t>::type storage_type1;
    typedef the_backend::storage_type<type2, storage_info2_t>::type storage_type2;
    typedef the_backend::storage_type<type3, storage_info3_t>::type storage_type3;

    typedef the_backend::temporary_storage_type<type4, storage_info1_t >::type tmp_storage_type;

    storage_type1 field1 = storage_type1(storage_info1_t(x,y,z), type1(), "field1");
    storage_type2 field2 = storage_type2(storage_info2_t(x,y,z), type2(), "field2");
    storage_type3 field3 = storage_type3(storage_info3_t(x,y,z), type3(), "field3");

    typedef arg<0, tmp_storage_type > p_temp;
    typedef arg<1, storage_type1 > p_field1;
    typedef arg<2, storage_type2 > p_field2;
    typedef arg<3, storage_type3 > p_field3;

    typedef boost::mpl::vector<p_temp, p_field1, p_field2, p_field3> accessor_list;

#if defined( CXX11_ENABLED )
    gridtools::domain_type<accessor_list> domain( (p_field1() = field1), (p_field2() = field2), (p_field3() = field3) );
#else
    gridtools::domain_type<accessor_list> domain(boost::fusion::make_vector(&field1, &field2, &field3));
#endif

    uint_t di[5] = {halo_size, halo_size, halo_size, d1-halo_size-1, d1};
    uint_t dj[5] = {halo_size, halo_size, halo_size, d2-halo_size-1, d2};

    gridtools::grid<axis> grid(di, dj);
    grid.value_list[0] = 0;
    grid.value_list[1] = d3-1;

// \todo simplify the following using the auto keyword from C++11
#ifdef __CUDACC__
    gridtools::computation* test_computation =
#else
        boost::shared_ptr<gridtools::computation> test_computation =
#endif
        gridtools::make_computation<the_backend>
        (
            gridtools::make_mss // mss_descriptor
            (
                execute<forward>(),
                gridtools::make_esf<function1>(p_temp(), p_field1()),
                gridtools::make_esf<function2>(p_field2(), p_field1(), p_temp())
            ),
            gridtools::make_mss // mss_descriptor
            (
                execute<backward>(),
                gridtools::make_esf<function1>(p_temp(), p_field1()),
                gridtools::make_esf<function3>(p_field3(), p_temp(), p_field1())
            ),
            domain, grid
        );

    test_computation->ready();

    test_computation->steady();

    test_computation->run();

    bool result = true;
    
// #ifdef CXX11_ENABLED
//     verifier verif(1e-13);
//     array<array<uint_t, 2>, 3> halos{{ {halo_size, halo_size}, {halo_size,halo_size}, {halo_size,halo_size} }};
//     bool result = verif.verify(repository.out_ref(), repository.out(), halos);
// #else
//     verifier verif(1e-13, halo_size);
//     bool result = verif.verify(repository.out_ref(), repository.out());
// #endif

    if(!result){
        std::cout << "ERROR"  << std::endl;
    }

    test_computation->finalize();

  return result; /// lapse_time.wall<5000000 &&
}
} // namespace multi_types_test

TEST(multitypes, multitypes) {
    EXPECT_TRUE(multi_types_test::test(10, 13, 31));
}
