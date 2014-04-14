#include <iostream>
#include "gt_assert.h"
#include "make_stencils.h"
#include "arg_type.h"
#include "storage.h"
#include "execution_types.h"
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/vector.hpp>
#include "range.h"
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/lambda.hpp>
#include "intermediate.h"

using namespace gridtools;

typedef int x_all;

struct lap_function {
    static const int n_args = 2;
    typedef arg_type<0> out;
    typedef const arg_type<1, range<-1, 1, -1, 1> > in;
    typedef boost::mpl::vector<out, in> arg_list;

    template <typename Domain>
    static void Do(Domain const & dom, x_all) {
        dom(out()) = 4*dom(in()) - 
            (dom(in( 1, 0, 0)) + dom(in( 0, 1, 0)) +
             dom(in(-1, 0, 0)) + dom(in( 0,-1, 0)));
    }
};

struct flx_function {
    static const int n_args = 3;
    typedef arg_type<0> out;
    typedef const arg_type<1, range<0, 1, 0, 0> > in;
    typedef const arg_type<2, range<0, 1, 0, 0> > lap;

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
    static const int n_args = 3;
    typedef arg_type<0> out;
    typedef const arg_type<1, range<0, 0, 0, 1> > in;
    typedef const arg_type<2, range<0, 0, 0, 1> > lap;
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
    static const int n_args = 5;
    typedef arg_type<0> out;
    typedef const arg_type<1> in;
    typedef const arg_type<2, range<-1, 0, 0, 0> > flx;
    typedef const arg_type<3, range<0, 0, -1, 0> > fly;
    typedef const arg_type<4> coeff;
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


struct print {
    std::string prefix_;

    print()
        :prefix_("")
    {}

    print(std::string const &s)
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
        boost::mpl::for_each<MplVector>(print(prefix_ + std::string("    ")));
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

    template <int I, int J, int K, int L>
    void operator()(range<I,J,K,L> const&) const {
        std::cout << prefix << range<I,J,K,L>() << std::endl;
    }

    template <typename MplVector>
    void operator()(MplVector const&) const {
        std::cout << "Independent" << std::endl;
        boost::mpl::for_each<MplVector>(print_(std::string("    ")));
        std::cout << "End Independent" << std::endl;
    }

    template <typename MplVector>
    void operator()(_impl::wrap_type<MplVector> const&) const {
        std::cout << "Independent" << std::endl;
        boost::mpl::for_each<MplVector>(print_(std::string("    ")));
        std::cout << "End Independent" << std::endl;
    }
};

template<typename MSS>
void print_mss(MSS)
{
    boost::mpl::for_each<typename MSS::linear_esf>(print(std::string(">")));

    std::cout << std::endl;

    boost::mpl::for_each<typename MSS::esf_array>(print(std::string(">")));

    std::cout << std::endl;

    typedef boost::mpl::fold<
        typename MSS::esf_array,
        boost::mpl::vector<>,
        _impl::traverse_ranges<boost::mpl::_1, boost::mpl::_2>
    >::type ranges_list;

    boost::mpl::for_each<ranges_list>(print_());

    std::cout << std::endl;

	typedef _impl::prefix_on_ranges<ranges_list>::type prefix_ranges;

    // typedef typename boost::mpl::fold<
    //     ranges_list,
    //     boost::mpl::vector<>,
    //     prefix_on_ranges<boost::mpl::_1,boost::mpl::_2>
    //     >::type ranges_list;

    boost::mpl::for_each<prefix_ranges>(print_());

    std::cout << std::endl;
}

int main() {
    typedef storage<double, gridtools::layout_map<0,1,2> > storage_type;

    typedef arg<5, storage_type > p_lap;
    typedef arg<4, storage_type > p_flx;
    typedef arg<3, storage_type > p_fly;
    typedef arg<2, storage_type > p_coeff;
    typedef arg<1, storage_type > p_in;
    typedef arg<0, storage_type > p_out;

    auto mss=make_mss(execute_upward, 
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
