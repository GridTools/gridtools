#include <storage.h>
#include <array.h>
#include <layout_map.h>
#include <axis.h>
#include <make_stencils.h>
#include <arg_type.h>
#include <execution_types.h>
#include <domain_type.h>
#include <boost/fusion/include/for_each.hpp>
#include <intermediate.h>
#include <backend_block.h>
#include <backend_naive.h>

typedef Interval<Level<0,-1>, Level<1,-1> > x_lap;
typedef Interval<Level<0,-1>, Level<1,-1> > x_flx;
typedef Interval<Level<0,-1>, Level<1,-1> > x_out;

typedef Interval<Level<0,-2>, Level<1,3> > axis;
//typedef extend_by<tight_axis, 2>::type axis;

struct lap_function {
    static const int n_args = 2;
    typedef arg_type<0> out;
    typedef const arg_type<1, range<-1, 1, -1, 1> > in;
    typedef typename boost::mpl::vector<out, in> arg_list;

    template <typename t_domain>
    static void Do(t_domain const & dom, x_lap) {
        dom(out()) = 3*dom(in()) - 
            (dom(in( 1, 0, 0)) + dom(in( 0, 1, 0)) +
             dom(in(-1, 0, 0)) + dom(in( 0,-1, 0)));
        std::cout << "*" << dom(out()) << "*" << std::endl;
    }
};

struct flx_function {
    static const int n_args = 3;
    typedef arg_type<0> out;
    typedef const arg_type<1, range<0, 1, 0, 0> > in;
    typedef const arg_type<2, range<0, 1, 0, 0> > lap;

    typedef typename boost::mpl::vector<out, in, lap> arg_list;

    template <typename t_domain>
    static void Do(t_domain const & dom, x_flx) {
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
    static void Do(t_domain const & dom, x_flx) {
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
    static void Do(t_domain const & dom, x_out) {
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
    template <typename T>
    void operator()(T const t) const {
        t->text();
    }
};

struct stampa {
    template <typename T>
    void operator()(T const t) const {
        std::cout << t << " ";
    }
};

struct stampa1 {
    template <typename T>
    void operator()(T const& t) const {
        std::cout << *t << " ";
    }
};

int main(int argc, char** argv) {
    int d1 = atoi(argv[1]);
    int d2 = atoi(argv[2]);
    int d3 = atoi(argv[3]);

    typedef storage<double, GCL::layout_map<0,1,2> > storage_type;

    storage_type in(d1,d2,d3,-1, std::string("in"));
    storage_type out(d1,d2,d3,-7.3, std::string("out"));
    storage_type coeff(d1,d2,d3,-3, std::string("coeff"));
    storage_type lap(d1,d2,d3,-4, std::string("lap"));
    storage_type flx(d1,d2,d3,-5, std::string("flx"));
    storage_type fly(d1,d2,d3,-6, std::string("fly"));

    out.print();

    GCL::array<storage_type*, 2> args;

    typedef arg<5, temporary<double> > p_lap;
    typedef arg<4, temporary<double> > p_flx;
    typedef arg<3, temporary<double> > p_fly;
    typedef arg<2, storage_type > p_coeff;
    typedef arg<1, storage_type > p_in;
    typedef arg<0, storage_type > p_out;

    typedef boost::mpl::vector<p_lap, p_flx, p_fly, p_coeff, p_in, p_out> arg_type_list;

    domain_type<arg_type_list> domain
        (boost::fusion::make_vector(&out, &in, &coeff /*,&fly, &flx*/));

    coordinates<axis> coords(2,d1-2,2,d2-2);
    coords.value_list[0] = 0;
    coords.value_list[1] = d3;

    // for (int i=0; i<d1; ++i)
    //     for (int j=0; j<d2; ++j) {
    //         domain.move_to(i,j,0);
    //         for (int k=0; k<d3; ++k) {
    //             domain[p_in()] = 1.0*(i-j*k);
    //             domain[p_coeff()] = 3.0*(i*j+k);
    //             domain.increment_along<2>();
    //         }
    //     }

    //in.print();

//    auto mss = make_mss(execute_upward, 
//                        make_esf<lap_function>(p_lap(), p_in()),
//                        make_esf<flx_function>(p_flx(), p_in(), p_lap()),
//                        make_esf<fly_function>(p_fly(), p_in(), p_lap()),
//                        make_esf<out_function>(p_out(), p_in(), p_flx(), p_fly(), p_coeff())
//                        );

#ifdef BACKEND_BLOCK
#define BACKEND backend_block
#else
#define BACKEND backend_naive
#endif

    intermediate::run<BACKEND>(make_mss(execute_upward, 
                                              make_esf<lap_function>(p_lap(), p_in()),
                                              make_independent(
                                                               make_esf<flx_function>(p_flx(), p_in(), p_lap()),
                                                               make_esf<fly_function>(p_fly(), p_in(), p_lap())
                                                               ),
                                              make_esf<out_function>(p_out(), p_in(), p_flx(), p_fly(), p_coeff())
                                              ), 
                                     domain, coords);


    in.print();
    out.print();
    //    lap.print();

    return 0;
}

