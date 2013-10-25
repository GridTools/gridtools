#include "storage.h"
#include "array.h"
#include "layout_map.h"
#include "axis.h"
#include "make_stencils.h"
#include "arg_type.h"
#include "execution_types.h"
#include "domain_type.h"
#include <boost/fusion/include/for_each.hpp>

typedef make_axis<Level<0,1>, Level<2,2> >::type axis;
typedef axis x_all;

struct lap_function {
    static const int n_args = 2;
    typedef arg_type<0> out;
    typedef const arg_type<1> in;

    template <typename t_domain>
    void operator()(t_domain const & dom, x_all) const {
        dom(out()) = 4*dom(in()) - 
            (dom(in( 1, 0, 0)) + dom(in( 0, 1, 0)) +
             dom(in(-1, 0, 0)) + dom(in( 0,-1, 0)));
    }
};

struct flx_function {
    static const int n_args = 3;
    typedef arg_type<0> out;
    typedef const arg_type<1> in;
    typedef const arg_type<2> lap;

    template <typename t_domain>
    void operator()(t_domain const & dom, x_all) const {
        dom(out()) = dom(lap(1,0,0))-dom(lap(0,0,0));
        if (dom(out())*(dom(in(1,0,0))-dom(in(0,0,0)))) {
            dom(out()) = 0;
        }
    }
};

struct fly_function {
    static const int n_args = 3;
    typedef arg_type<0> out;
    typedef const arg_type<1> in;
    typedef const arg_type<2> lap;

    template <typename t_domain>
    void operator()(t_domain const & dom, x_all) const {
        dom(out()) = dom(lap(0,1,0))-dom(lap(0,0,0));
        if (dom(out())*(dom(in(0,1,0))-dom(in(0,0,0)))) {
            dom(out()) = 0;
        }
    }
};

struct out_function {
    static const int n_args = 3;
    typedef arg_type<0> out;
    typedef const arg_type<1> in;
    typedef const arg_type<3> flx;
    typedef const arg_type<4> fly;
    typedef const arg_type<5> coeff;

    template <typename t_domain>
    void operator()(t_domain const & dom, x_all) const {
        dom(out()) = dom(in()) - dom(coeff()) *
            (dom(flx()) - dom(flx(-1,0,0)) +
             dom(fly()) - dom(fly(0,-1,0))
             );
    }
};

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

    typedef storage<int, GCL::layout_map<0,1,2> > storage_type;

    storage_type in(d1,d2,d3,2);
    storage_type out(d1,d2,d3,1);
    storage_type coeff(d1,d2,d3,1);

    GCL::array<storage_type*, 2> args;

    typedef arg<5, temporary<int> > p_lap;
    typedef arg<4, temporary<int> > p_flx;
    typedef arg<3, temporary<int> > p_fly;
    typedef arg<2, storage_type > p_coeff;
    typedef arg<1, storage_type > p_in;
    typedef arg<0, storage_type > p_out;

    typedef boost::mpl::vector<p_lap, p_flx, p_fly, p_coeff, p_in, p_out> arg_type_list;

//     domain_type<iteration_space,
//         boost::mpl::vector<p_lap, p_flx, p_ply, p_coeff, p_in, p_out> >
//         init_domain(args,
//                     iteration_space(d1,d2,d3));

    domain_type<arg_type_list> domain
        (boost::fusion::vector<storage_type*,storage_type*,storage_type*>(&out, &in, &coeff));

    { // JUST CHECKING FEW THINGS
        boost::mpl::for_each<domain_type<arg_type_list>::arg_list>(print());
        boost::fusion::for_each(domain_type<arg_type_list>().args, print());
        boost::fusion::for_each(domain_type<arg_type_list>::raw_index_list(), stampa());
        std::cout << std::endl;
        //domain_type<arg_type_list>::arg_list_view lv(domain_type<arg_type_list>::raw_index_list());
        //boost::fusion::for_each(lv, stampa1());
        std::cout << "OCIO" << std::endl;
        domain_type<arg_type_list>::arg_list_view listview(domain.args);
        boost::fusion::for_each(listview, print());
        std::cout << std::endl;
        std::cout << "OCIO1" << std::endl;
        boost::fusion::for_each(domain_type<arg_type_list>::arg_list_view(domain.args), print());
        std::cout << std::endl;
        std::cout << "OCIO2" << std::endl;
        domain_type<arg_type_list>::arg_list args2;
        domain_type<arg_type_list>::arg_list_view listview2(args2);
        boost::fusion::for_each(listview2, print());
        std::cout << std::endl;
    }

    coordinates<axis> coords;
    coords.value_list[0] = 2;
    coords.value_list[1] = d1/2;
    coords.value_list[2] = d1;

    auto mss = make_mss(execute_upward, 
                        make_esf<lap_function>(p_lap(), p_in()),
                        make_esf<flx_function>(p_flx(), p_in(), p_lap()),
                        make_esf<fly_function>(p_fly(), p_in(), p_lap()),
                        make_esf<out_function>(p_out(), p_in(), p_flx(), p_fly(), p_coeff())
                        );

    //prepare_argumens<arg_type_list>(coeff, in, out);

    domain.setup_temporaries(coords, mss); // To put the storage information for the temporary stages

    return 0;
}

// int main(int argc, char** argv) {


//     int d1 = atoi(argv[1]);
//     int d2 = atoi(argv[2]);
//     int d3 = atoi(argv[3]);

//     typedef storage<int, GCL::layout_map<0,1,2> > storage_type;

//     storage_type in(d1,d2,d3,2);
//     storage_type out(d1,d2,d3,1);

//     GCL::array<storage_type*, 2> args;

//     args[0] = &out;
//     args[1] = &in;

//     typedef arg<0, storage_type* > p_lap;
//     typedef arg<1, storage_type* > p_in;

//     domain_type<iteration_space,
//                     boost::mpl::vector<p_in,p_lap>, 
//                     storage_type* > 
//         init_domain(args, 
//                     iteration_space(d1,d2,d3));

//     backend0::run( make_mss(GCL::layout_map<0,1,2>(), 
//                                     utils::boollist<forward, forward, forward>(), 
//                                     make_esf<init>(p_lap(), p_in())),
//                        init_domain );


//     std::cout << "\nin" << std::endl;
//     in.print();
//     std::cout << "out" << std::endl;
//     out.print();


//     domain_type<iteration_space,
//                      boost::mpl::vector<p_in,p_lap>, 
//                      storage_type* > 
//         compute_domain(args, 
//                        iteration_space(1,1,1,d1-1,d2-1,d3-1));

//     backend0::run( make_mss(GCL::layout_map<0,1,2>(), 
//                                     utils::boollist<forward, forward, forward>(), 
//                                     make_esf<lap_function>(p_lap(), p_in())),
//                        compute_domain );


//     std::cout << "------------------------------------------" << std::endl;
//     std::cout << "in" << std::endl;
//     in.print();
//     std::cout << "\nout" << std::endl;
//     out.print();

//     return 0;
// }
