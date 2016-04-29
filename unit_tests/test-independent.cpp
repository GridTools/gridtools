#include <iostream>
#include "common/host_device.hpp"
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/lambda.hpp>
#include "common/gt_assert.hpp"
#include "stencil-composition/stencil-composition.hpp"

using namespace gridtools;
using namespace enumtype;

typedef uint_t x_all;

struct lap_function {
    typedef accessor<0, enumtype::inout> out;
    typedef accessor<1, enumtype::in, extent<-1, 1, -1, 1> > in;
    typedef boost::mpl::vector<out, in> arg_list;

    template <typename Domain>
    static void Do(Domain const & dom, x_all) {
        dom(out()) = 4*dom(in()) -
            (dom(in( 1, 0, 0)) + dom(in( 0, 1, 0)) +
             dom(in(-1, 0, 0)) + dom(in( 0,-1, 0)));
    }
};

struct flx_function {
    typedef accessor<0, enumtype::inout> out;
    typedef accessor<1, enumtype::in, extent<0, 1, 0, 0> > in;
    typedef accessor<2, enumtype::in, extent<0, 1, 0, 0> > lap;
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
    typedef accessor<0, enumtype::inout> out;
    typedef accessor<1, enumtype::in, extent<0, 0, 0, 1> > in;
    typedef accessor<2, enumtype::in,  extent<0, 0, 0, 1> > lap;
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
    typedef accessor<0> out;
    typedef accessor<1> in;
    typedef accessor<2, enumtype::in, extent<-1, 0, 0, 0> > flx;
    typedef accessor<3, enumtype::in, extent<0, 0, -1, 0> > fly;
    typedef accessor<4> coeff;

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
    void operator()(extent<I,J,K,L> const&) const {
        std::cout << prefix << extent<I,J,K,L>() << std::endl;
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
    typedef typename mss_descriptor_linear_esf_sequence<MSS>::type linear_esf_t;

    boost::mpl::for_each<linear_esf_t>(print_independent(std::string(">")));

    std::cout << std::endl;

    typedef typename mss_descriptor_esf_sequence<MSS>::type esf_sequence_t;

    boost::mpl::for_each<esf_sequence_t>(print_independent(std::string(">")));

    std::cout << std::endl;

    typedef typename boost::mpl::fold<
        esf_sequence_t,
        boost::mpl::vector<>,
        strgrid::traverse_extents<boost::mpl::_1, boost::mpl::_2>
    >::type extents_list;
    boost::mpl::for_each<extents_list>(print_());

    std::cout << std::endl;

    typedef typename strgrid::prefix_on_extents<extents_list>::type prefix_extents;

    // typedef typename boost::mpl::fold<
    //     extents_list,
    //     boost::mpl::vector<>,
    //     prefix_on_extents<boost::mpl::_1,boost::mpl::_2>
    //     >::type extents_list;

    boost::mpl::for_each<prefix_extents>(print_());

    std::cout << std::endl;
}

int main() {

    typedef base_storage< wrap_pointer< float_type >,
        backend< Host, structured, Naive >::storage_info< 0, gridtools::layout_map< 0, 1, 2 > >,
        1 > storage_type;

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
