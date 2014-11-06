
#include <gridtools.h>
#ifdef CUDA_EXAMPLE
#include <stencil-composition/backend_cuda.h>
#else
#include <stencil-composition/backend_host.h>
#endif
//#include "tridiagonal.h"


namespace vertical_advection{

using namespace gridtools;
using namespace enumtype;

//################################################################################
//                              Finite Differences
//################################################################################

    template <typename Pointer>
    struct smart : public Pointer{

        explicit smart(typename Pointer::pointee_t* p) : Pointer(p), m_owner(true) {}
        explicit smart(uint_t size) : Pointer(size), m_owner(true) {}

// constructor from base class defines
        __device__ __host__
        explicit smart(smart<Pointer> const& other)
            : m_owner(other.owner? true : false )//if the other is not owner I cannot take ownership
        {
            other.m_owner=false;
        }
        void free_it(){if(m_owner){ Pointer::free_it();}}//if I'm not the owner I don't free the pointers

    private:
        bool m_owner;
    };


    // template < typename Storage, ushort_t AccuracyOrder>
    // ushort_t integrator<Storage, AccuracyOrder>::m_lru=0;





//############################################################################
//                            TEST
//############################################################################




    // template <ushort_t AccuracyOrder, ushort_t DerivativeOrder>
    // struct BDF{
    // };


    // template <typename ArgType1>
    // struct expr_bdf public expr<ArgType1, ArgType2 >{
    //     typedef expr<ArgType1, ArgType2> super;
    //     GT_FUNCTION
    //     expr_bdf(ArgType1 const& first_operand):first_operand(first_operand){}
    //     ArgType1 const first_operand;
    // };

    // template <typename ArgType1>
    // GT_FUNCTION
    // auto inline value(expr_bdf<ArgType1> const& arg) const -> decltype((*this)(arg.first_operand.solutions(0))) {return (*this)(arg.first_operand.solutions(0));}

    // template<typename FieldType, ushort_t AccuracyOrder, ushort_t DerivationOrder>
    // auto inline diff(integrator<FieldType, AccuracyOrder>){}
    //     auto lhs(){}

    // template<typename FieldType, short_t AccuracyOrder, short_t DerivativeOrder>
    // struct


    // tmeplate<AccuracyOrder>
    // struct beta_method

    // template <typename Scheme, typename Expr...>
    //     struct BDF{

    //         return BDF;
    //     };

// This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval<level<0,-1>, level<1,1> > x_lap;
    typedef gridtools::interval<level<0,-1>, level<1,2> > axis;

    using gridtools::arg_type;
    using namespace enumtype;
    using namespace expressions;

    struct functor {
        typedef  arg_type<0> out;
        typedef  arg_decorator<  arg_type< 1 > > in;//develop this with recursion + index
        typedef boost::mpl::vector<out, in> arg_list;
        using time=Extra<1>;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_lap) {
            //todo: prevent this: dom(out()) = dom(in(Extra<1>(1))); when there are no extra dimensions
            eval(out()) = (eval(in(time(-1)) )+eval(in()));/*-in(time(0))) )/2;*/
    }
    };

/*
 * The following operators and structs are for debugging only
 */
    std::ostream& operator<<(std::ostream& s, functor const) {
        return s << "functor";
    }

    bool test(int x, int y, int z){

        // time_integrator<double, &value_pointer> integrator;

        // printf("%f\n", integrator.getField(0));
        typedef gridtools::layout_map<0,1,2> layout_t;
        typedef gridtools::backend<Host, Naive >::storage_type<float_type, layout_t >::type storage_type;
        typedef extend_width<storage_type, 2> extended_type;
        typedef extend_dim<extended_type> integrator_type;

        storage_type out(10,10,3,4., std::string("out"));
        integrator_type in(10,10,3,1., std::string("in"));;
        storage_type init(10,10,3,2., std::string("init"));

        //initialization
        in.push_back(init.data());//using the same data pointer might generate hazards!

        in.print();

        typedef arg<0, storage_type > p_out;
        typedef arg<1, integrator_type > p_in;
        typedef boost::mpl::vector<p_out, p_in> arg_type_list;

        gridtools::domain_type<arg_type_list> domain( (p_out() = out), (p_in() = in) );

        uint_t di[5] = {1, 0, 0, 10-1, 10};
        //uint_t dj[5] = {0, 0, 0, 10, 10};
        gridtools::coordinates<axis> coords(di, di);
        coords.value_list[0] = 0;
        coords.value_list[1] = 1;

        auto test =  gridtools::make_computation<gridtools::backend<Host, Naive >, layout_t>
            (
                gridtools::make_mss // mss_descriptor
                (
                    execute<parallel>(),
                    gridtools::make_esf<functor>(p_out(), p_in()) ),
                domain, coords
                );

            test->ready();
            test->steady();
            domain.clone_to_gpu();
            //for(ushort_t i=0; i<2; ++i){
            test->run();
            in.push_back(out.data());
            in.print();
            out.print();
            //}
            test->run();
            in.print();
            out.print();

        return true;

    };


}//namespace time_integration
