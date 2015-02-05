#include <iostream>
#include <common/host_device.h>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/lambda.hpp>
#include <common/gt_assert.h>
#include <stencil-composition/make_stencils.h>
#include <stencil-composition/arg_type.h>
#include <stencil-composition/execution_types.h>
#include <stencil-composition/range.h>
#include <stencil-composition/intermediate.h>

using namespace gridtools;
using namespace enumtype;

typedef uint_t x_all;

struct lap_function {
#ifdef CXX11_ENABLED
    typedef arg_type<0> out;
    typedef const arg_type<1, range<-1, 1, -1, 1> > in;
#else
    typedef arg_type<0>::type out;
    typedef const arg_type<1, range<-1, 1, -1, 1> >::type in;
#endif
    typedef boost::mpl::vector<out, in> arg_list;

    template <typename Domain>
    static void Do(Domain const & dom, x_all) {
        dom(out()) = 4*dom(in()) -
            (dom(in( 1, 0, 0)) + dom(in( 0, 1, 0)) +
             dom(in(-1, 0, 0)) + dom(in( 0,-1, 0)));
    }
};

struct flx_function {
#ifdef CXX11_ENABLED
    typedef arg_type<0> out;
    typedef const arg_type<1, range<0, 1, 0, 0> > in;
    typedef const arg_type<2, range<0, 1, 0, 0> > lap;
#else
    typedef arg_type<0>::type out;
    typedef const arg_type<1, range<0, 1, 0, 0> >::type in;
    typedef const arg_type<2, range<0, 1, 0, 0> >::type lap;
#endif
    typedef boost::mpl::vector<out, in, lap> arg_list;

    template <typename Domain>
    static void Do(Domain const & dom, x_all) {
        dom(out()) = dom(lap(1,0,0))-dom(lap(0,0,0));
        if (dom(out())*(dom(in(1,0,0))-dom(in(0,0,0)))) {
            dom(out()) = 0.;
        }
    }
};

struct fly_function {
#ifdef CXX11_ENABLED
    typedef arg_type<0> out;
    typedef const arg_type<1, range<0, 0, 0, 1> > in;
    typedef const arg_type<2, range<0, 0, 0, 1> > lap;
#else
    typedef arg_type<0>::type out;
    typedef const arg_type<1, range<0, 0, 0, 1> >::type in;
    typedef const arg_type<2, range<0, 0, 0, 1> >::type lap;
#endif
    typedef boost::mpl::vector<out, in, lap> arg_list;

    template <typename Domain>
    static void Do(Domain const & dom, x_all) {
        dom(out()) = dom(lap(0,1,0))-dom(lap(0,0,0));
        if (dom(out())*(dom(in(0,1,0))-dom(in(0,0,0)))) {
            dom(out()) = 0.;
        }
    }
};

struct out_function {
#ifdef CXX11_ENABLED
    typedef arg_type<0> out;
    typedef const arg_type<1> in;
    typedef const arg_type<2, range<-1, 0, 0, 0> > flx;
    typedef const arg_type<3, range<0, 0, -1, 0> > fly;
    typedef const arg_type<4> coeff;
#else
    typedef arg_type<0>::type out;
    typedef const arg_type<1>::type in;
    typedef const arg_type<2, range<-1, 0, 0, 0> >::type flx;
    typedef const arg_type<3, range<0, 0, -1, 0> >::type fly;
    typedef const arg_type<4>::type coeff;
#endif

    typedef boost::mpl::vector<out,in,flx,fly,coeff> arg_list;

    template <typename Domain>
    static void Do(Domain const & dom, x_all) {
        dom(out()) = dom(in()) - dom(coeff()) *
            (dom(flx()) - dom(flx(-1,0,0)) +
             dom(fly()) - dom(fly(0,-1,0))
             );
    }
};

std::ostream& operator<<(std::ostream& s, lap_function const) {
    return s << "lap_function";
}
std::ostream& operator<<(std::ostream& s, flx_function const) {
    return s << "flx_function";
}
std::ostream& operator<<(std::ostream& s, fly_function const) {
    return s << "fly_function";
}
std::ostream& operator<<(std::ostream& s, out_function const) {
    return s << "out_function";
}


struct print_independent {
    std::string prefix_;

    print_independent()
        :prefix_("")
    {}

    print_independent(std::string const &s)
        : prefix_(s)
    {}

    template <typename T>
    void operator()(T const) const {
        std::cout << "*";
        std::cout << prefix_ << typename T::esf_function() << std::endl;
    }

    template <typename MplVector>
    void operator()(independent_esf<MplVector> const&) const {
        std::cout << "Independent" << std::endl;
        boost::mpl::for_each<MplVector>(print_independent(prefix_ + std::string("    ")));
        std::cout << "End Independent" << std::endl;
    }
};

struct print_ {
    std::string prefix;

    print_()
        :prefix("")
    {}

    print_(std::string const &s)
        : prefix(s)
    {}

    template <uint_t I, uint_t J, uint_t K, uint_t L>
    void operator()(range<I,J,K,L> const&) const {
        std::cout << prefix << range<I,J,K,L>() << std::endl;
    }

    template <typename MplVector>
    void operator()(MplVector const&) const {
        std::cout << "Independent" << std::endl;
        //boost::mpl::for_each<MplVector>(print_(std::string("    ")));
        std::cout << "End Independent" << std::endl;
    }

    template <typename MplVector>
    void operator()(_impl::wrap_type<MplVector> const&) const {
        std::cout << "Independent" << std::endl;
        //boost::mpl::for_each<MplVector>(print_(std::string("    ")));
        std::cout << "End Independent" << std::endl;
    }
};

template<typename MSS>
void print_mss(MSS)
{
    boost::mpl::for_each<typename MSS::linear_esf>(print_independent(std::string(">")));

    std::cout << std::endl;

    boost::mpl::for_each<typename MSS::esf_array>(print_independent(std::string(">")));

    std::cout << std::endl;

    typedef typename boost::mpl::fold<
        typename MSS::esf_array,
        boost::mpl::vector<>,
        _impl::traverse_ranges<boost::mpl::_1, boost::mpl::_2>
    >::type ranges_list;
    boost::mpl::for_each<ranges_list>(print_());

    std::cout << std::endl;

	typedef typename _impl::prefix_on_ranges<ranges_list>::type prefix_ranges;

    // typedef typename boost::mpl::fold<
    //     ranges_list,
    //     boost::mpl::vector<>,
    //     prefix_on_ranges<boost::mpl::_1,boost::mpl::_2>
    //     >::type ranges_list;

    boost::mpl::for_each<prefix_ranges>(print_());

    std::cout << std::endl;
}

int main() {
    typedef base_storage<enumtype::Host, float_type, gridtools::layout_map<0,1,2> > storage_type;

    typedef arg<5, storage_type > p_lap;
    typedef arg<4, storage_type > p_flx;
    typedef arg<3, storage_type > p_fly;
    typedef arg<2, storage_type > p_coeff;
    typedef arg<1, storage_type > p_in;
    typedef arg<0, storage_type > p_out;

    auto mss=make_mss(execute<forward>(),
                      make_esf<lap_function>(p_lap(), p_in()),
                      make_independent(
                                       make_esf<flx_function>(p_flx(), p_in(), p_lap()),
                                       make_esf<fly_function>(p_fly(), p_in(), p_lap())
                                       ),
                      make_esf<out_function>(p_out(), p_in(), p_flx(), p_fly(), p_coeff())
                      );

    print_mss(mss);

    return 0;
}
