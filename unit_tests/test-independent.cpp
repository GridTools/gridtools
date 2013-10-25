#include <iostream>
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

// template <typename t_functor_desc>
// struct extract_ranges {
//     typedef typename t_functor_desc::esf_function t_functor;

//     template <typename range_state, typename argument_index>
//     struct update_range {
//         typedef typename boost::mpl::at<typename t_functor::arg_list, argument_index>::type argument_type;
//         typedef typename enclosing_range<range_state, typename argument_type::range_type>::type type;
//     };

//     typedef typename boost::mpl::fold<
//         boost::mpl::range_c<int, 0, t_functor::n_args>,
//         range<0,0,0,0>,
//         update_range<boost::mpl::_1, boost::mpl::_2>
//         >::type type;
// };

// template <typename T>
// struct extract_ranges<independent_esf<T> >
// {
//     typedef typename boost::false_type type;
// };

// template <typename not_independent_elem>
// struct from_independents {
//     typedef typename boost::false_type type;
// };

// template <typename mpl_array>
// struct wrap_type {
//     typedef mpl_array type;
// };

// template <typename T>
// struct from_independents<independent_esf<T> > {
//     typedef typename boost::mpl::fold<
//         typename independent_esf<T>::esf_list,
//         boost::mpl::vector<>,
//         boost::mpl::push_back<boost::mpl::_1, extract_ranges<boost::mpl::_2> >
//         >::type raw_type;

//     typedef wrap_type<raw_type> type;
// };

// template <typename state, typename elem>
// struct traverse_ranges {

//     typedef typename boost::mpl::push_back<
//         state,
//         typename boost::mpl::if_<
//             is_independent<elem>,
//             typename from_independents<elem>::type,
//             typename extract_ranges<elem>::type
//             >::type
//             >::type type;
// };


// template <typename list_of_ranges>
// struct prefix_on_ranges {

//     template <typename t_list, typename t_range>
//     struct state {
//         typedef t_list list;
//         typedef t_range range;
//     };

//     template <typename previous_state, typename current_element>
//     struct update_state {
//         typedef typename sum_range<typename previous_state::range,
//                                    current_element>::type new_range;
//         typedef typename boost::mpl::push_front<typename previous_state::list, new_range>::type new_list;
//         typedef state<new_list, new_range> type;
//     };

//     template <typename previous_state, typename ind_vector>
//     struct update_state<previous_state, wrap_type<ind_vector> > {
//         typedef typename boost::mpl::fold<
//             ind_vector,
//             boost::mpl::vector<>,
//             boost::mpl::push_back<boost::mpl::_1, sum_range<typename previous_state::range, boost::mpl::_2> >
//             >::type raw_ranges;

//         typedef typename boost::mpl::fold<
//             raw_ranges,
//             range<0,0,0,0>,
//             enclosing_range<boost::mpl::_2, boost::mpl::_1>
//             >::type final_range;

//         typedef typename boost::mpl::push_front<typename previous_state::list, wrap_type<raw_ranges> >::type new_list;

//         typedef state<new_list, final_range> type;
//     };

//     typedef typename boost::mpl::reverse_fold<
//         list_of_ranges,
//         state<boost::mpl::vector<>, range<0,0,0,0> >,
//         update_state<boost::mpl::_1, boost::mpl::_2> >::type final_state;

//      typedef typename final_state::list type;
// };




typedef int x_all;

struct lap_function {
    static const int n_args = 2;
    typedef arg_type<0> out;
    typedef const arg_type<1, range<-1, 1, -1, 1> > in;
    typedef typename boost::mpl::vector<out, in> arg_list;

    template <typename t_domain>
    static void Do(t_domain const & dom, x_all) {
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

    typedef typename boost::mpl::vector<out, in, lap> arg_list;

    template <typename t_domain>
    static void Do(t_domain const & dom, x_all) {
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
    typedef typename boost::mpl::vector<out, in, lap> arg_list;

    template <typename t_domain>
    static void Do(t_domain const & dom, x_all) {
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
    typedef typename boost::mpl::vector<out,in,flx,fly,coeff> arg_list;

    template <typename t_domain>
    static void Do(t_domain const & dom, x_all) {
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

    template <typename mplvec>
    void operator()(independent_esf<mplvec> const&) const {
        std::cout << "Independent" << std::endl;
        boost::mpl::for_each<mplvec>(print(prefix_ + std::string("    ")));
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

    template <int i, int j, int k, int l>
    void operator()(range<i,j,k,l> const&) const {
        std::cout << prefix << range<i,j,k,l>() << std::endl;
    }

    template <typename mplvec>
    void operator()(mplvec const&) const {
        std::cout << "Independent" << std::endl;
        boost::mpl::for_each<mplvec>(print_(std::string("    ")));
        std::cout << "End Independent" << std::endl;
    }

    template <typename mplvec>
    void operator()(_impl::wrap_type<mplvec> const&) const {
        std::cout << "Independent" << std::endl;
        boost::mpl::for_each<mplvec>(print_(std::string("    ")));
        std::cout << "End Independent" << std::endl;
    }
};

int main() {
    typedef storage<double, GCL::layout_map<0,1,2> > storage_type;

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

    boost::mpl::for_each<typename decltype(mss)::linear_esf>(print(std::string(">")));

    std::cout << std::endl;

    boost::mpl::for_each<typename decltype(mss)::esf_array>(print(std::string(">")));

    std::cout << std::endl;

    typedef typename boost::mpl::fold<
    typename decltype(mss)::esf_array,
        boost::mpl::vector<>,
        _impl::traverse_ranges<boost::mpl::_1,boost::mpl::_2>
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

    return 0;
}
