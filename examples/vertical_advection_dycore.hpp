#pragma once
#include <gridtools.hpp>

#include <stencil-composition/backend.hpp>
#include <boost/fusion/include/make_vector.hpp>

#include "vertical_advection_repository.hpp"
#include <tools/verifier.hpp>
#include <stencil-composition/make_computation.hpp>

/*
  This file shows an implementation of the "vertical advection" stencil used in COSMO for U field
 */

using gridtools::level;
using gridtools::accessor;
using gridtools::range;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

namespace vertical_advection_dycore{
// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval<level<0, 1>, level<1,-2> > kbody;
typedef gridtools::interval<level<0, -1>, level<1,-2> > kbody_low;
typedef gridtools::interval<level<0,-1>, level<0,-1> > kminimum;
typedef gridtools::interval<level<1,-1>, level<1,-1> > kmaximum;

typedef gridtools::interval<level<0,-1>, level<1,1> > axis;

template<typename T>
struct u_forward_function {
    typedef const accessor<0> utens_stage;
    typedef const accessor<1, range<0,1, 0, 0> > wcon;
    typedef const accessor<2> u_stage;
    typedef const accessor<3> u_pos;
    typedef const accessor<4> utens;
    typedef const accessor<5> dtr_stage;
    typedef accessor<6> acol;
    typedef accessor<7> bcol;
    typedef accessor<8> ccol;
    typedef accessor<9> dcol;

    typedef boost::mpl::vector<utens_stage, wcon, u_stage, u_pos, utens, dtr_stage, acol, bcol, ccol, dcol> arg_list;

    template<typename Eval>
    GT_FUNCTION
    static void Do(Eval const & eval, kbody interval)
    {
        //TODO use Average function here
        T gav = (T)-0.25 * (eval(wcon( 1, 0, 0)) + eval(wcon(0,0,0)));
        T gcv = (T)0.25 * (eval(wcon( 1, 0, 1)) + eval(wcon(0,0,1)));

        T as = gav * BET_M;
        T cs = gcv * BET_M;

        eval(acol()) = gav * BET_P;
        eval(ccol()) = gcv * BET_P;
        eval(bcol()) = eval(dtr_stage()) - eval(acol()) - eval(ccol());

        T correctionTerm = -as * (eval(u_stage(0,0,-1)) - eval(u_stage())) -
                cs * (eval(u_stage(0,0,1)) - eval(u_stage()));
        // update the d column
        computeDColumn(eval, correctionTerm);
        thomas_forward(eval, interval);
    }

    template<typename Eval>
    GT_FUNCTION
    static void Do(Eval const & eval, kmaximum interval)
    {
        T gav = -(T)0.25 * (eval(wcon(1,0,0)) + eval(wcon()));
        T as = gav * BET_M;

        eval(acol()) = gav * BET_P;
        eval(bcol()) = eval(dtr_stage()) - eval(acol());

        T correctionTerm = -as * (eval(u_stage(0,0,-1)) - eval(u_stage()));

        // update the d column
        computeDColumn(eval, correctionTerm);
        thomas_forward(eval, interval);
    }

    template<typename Eval>
    GT_FUNCTION
    static void Do(Eval const & eval, kminimum interval)
    {
        T gcv = (T)0.25 * (eval(wcon( 1, 0, 1)) + eval(wcon(0,0,1)));
        T cs = gcv * BET_M;

        eval(ccol()) = gcv * BET_P;
        eval(bcol()) = eval(dtr_stage()) - eval(ccol());

        T correctionTerm = -cs * (eval(u_stage(0,0,1)) - eval(u_stage()));
        // update the d column
        computeDColumn(eval, correctionTerm);
        thomas_forward(eval, interval);
    }

private:
    template<typename Eval>
    GT_FUNCTION
    static void computeDColumn(Eval const & eval, const T correctionTerm)
    {
        eval(dcol()) = eval(dtr_stage()) * eval(u_pos()) + eval(utens()) + eval(utens_stage()) + correctionTerm;
    }

    template <typename Eval>
    GT_FUNCTION
    static void thomas_forward(Eval const & eval, kbody) {
        T divided = (T)1.0 / (eval(bcol()) - (eval(ccol(0,0,-1)) * eval(acol())));
        eval(ccol()) = eval(ccol())* divided;
        eval(dcol()) = (eval(dcol()) - (eval(dcol(0,0,-1)) * eval(acol()))) * divided;
    }

    template <typename Eval>
    GT_FUNCTION
    static void thomas_forward(Eval const & eval, kmaximum) {
        T divided = (T)1.0 / (eval(bcol()) - eval(ccol(0,0,-1)) * eval(acol()));
        eval(dcol()) = (eval(dcol()) - eval(dcol(0,0,-1)) * eval(acol())) * divided;
    }

    template <typename Eval>
    GT_FUNCTION
    static void thomas_forward(Eval const & eval, kminimum)
    {
        T divided = (T)1.0 / eval(bcol());
        eval(ccol()) = eval(ccol()) * divided;
        eval(dcol()) = eval(dcol()) * divided;
    }
};

template<typename T>
struct u_backward_function {
    typedef accessor<0> utens_stage;
    typedef const accessor<1> u_pos;
    typedef const accessor<2> dtr_stage;
    typedef accessor<3> ccol;
    typedef accessor<4> dcol;
    typedef accessor<5> data_col;

    typedef boost::mpl::vector<utens_stage, u_pos, dtr_stage, ccol, dcol, data_col> arg_list;

    template<typename Eval>
    GT_FUNCTION
    static void Do(Eval const & eval, kbody_low interval)
    {
        eval(utens_stage()) = eval(dtr_stage()) * (thomas_backward(eval, interval) - eval(u_pos()));
    }

    template<typename Eval>
    GT_FUNCTION
    static void Do(Eval const & eval, kmaximum interval)
    {
        eval(utens_stage()) = eval(dtr_stage()) * (thomas_backward(eval, interval) - eval(u_pos()));
    }

private:
    template <typename Eval>
    GT_FUNCTION
    static T thomas_backward(Eval const & eval, kbody_low) {
        T datacol = eval(dcol()) - eval(ccol()) * eval(data_col(0,0,1));
        eval(data_col()) = datacol;
        return datacol;
    }

    template <typename Eval>
    GT_FUNCTION
    static T thomas_backward(Eval const & eval, kmaximum) {
        T datacol = eval(dcol());
        eval(data_col()) = datacol;
        return datacol;
    }
};

/*
 * The following operators and structs are for debugging only
 */
//std::ostream& operator<<(std::ostream& s, u_forward_function<double> const) {
//    return s << "u_forward_function";
//}
std::ostream& operator<<(std::ostream& s, u_backward_function<double> const) {
    return s << "u_backward_function";
}

bool test(uint_t d1, uint_t d2, uint_t d3) {

    const int halo_size = 3;

    typedef gridtools::layout_map<0,1,2> layout_ijk;
    typedef gridtools::layout_map<0> layout_scalar;


    typedef vertical_advection::repository::storage_type storage_type;
    typedef vertical_advection::repository::scalar_storage_type scalar_storage_type;
    typedef vertical_advection::repository::tmp_storage_type tmp_storage_type;

    vertical_advection::repository repository(d1, d2, d3, halo_size);
    repository.init_fields();

    repository.generate_reference();

    // Definition of placeholders. The order of them reflect the order the user will deal with them
    // especially the non-temporary ones, in the construction of the domain
    typedef arg<0, storage_type> p_utens_stage;
    typedef arg<1, storage_type> p_u_stage;
    typedef arg<2, storage_type> p_wcon;
    typedef arg<3, storage_type> p_u_pos;
    typedef arg<4, storage_type> p_utens;
    typedef arg<5, scalar_storage_type> p_dtr_stage;
    typedef arg<6, tmp_storage_type> p_acol;
    typedef arg<7, tmp_storage_type> p_bcol;
    typedef arg<8, tmp_storage_type> p_ccol;
    typedef arg<9, tmp_storage_type> p_dcol;
    typedef arg<10, tmp_storage_type> p_data_col;

    // An array of placeholders to be passed to the domain
    // I'm using mpl::vector, but the final API should look slightly simpler
    typedef boost::mpl::vector<p_utens_stage, p_u_stage, p_wcon, p_u_pos, p_utens, p_dtr_stage,
            p_acol, p_bcol, p_ccol, p_dcol, p_data_col> accessor_list;

    // construction of the domain. The domain is the physical domain of the problem, with all the physical fields that are used, temporary and not
    // It must be noted that the only fields to be passed to the constructor are the non-temporary.
    // The order in which they have to be passed is the order in which they appear scanning the placeholders in order. (I don't particularly like this)

#ifdef CXX11_ENABLE
    gridtools::domain_type<accessor_list> domain(
            (p_utens_stage() = repository.utens_stage()),
            (p_u_stage() = repository.u_stage()),
            (p_wcon() = repository.wcon()),
            (p_u_pos() = repository.u_pos()),
            (p_utens() = repository.utens()) ,
            (p_dtr_stage() = repository.dtr_stage())
    );
#else
    gridtools::domain_type<accessor_list> domain(boost::fusion::make_vector(
            &repository.utens_stage(), &repository.u_stage(), &repository.wcon(),
            &repository.u_pos(), &repository.utens(), &repository.dtr_stage()));

#endif

    // Definition of the physical dimensions of the problem.
    // The constructor takes the horizontal plane dimensions,
    // while the vertical ones are set according the the axis property soon after
    // gridtools::coordinates<axis> coords(2,d1-2,2,d2-2);
    uint_t di[5] = {halo_size, halo_size, halo_size, d1-halo_size-1, d1};
    uint_t dj[5] = {halo_size, halo_size, halo_size, d2-halo_size-1, d2};

    gridtools::coordinates<axis> coords(di, dj);
    coords.value_list[0] = 0;
    coords.value_list[1] = d3-1;

//todo simplify the following using the auto keyword from C++11
#ifdef __CUDACC__
    gridtools::computation* vertical_advection =
#else
        boost::shared_ptr<gridtools::computation> vertical_advection =
#endif
        gridtools::make_computation<vertical_advection::va_backend>
        (
            gridtools::make_mss // mss_descriptor
            (
                execute<forward>(),
                gridtools::make_esf<u_forward_function<double> >(
                        p_utens_stage(),
                        p_wcon(),
                        p_u_stage(),
                        p_u_pos(),
                        p_utens(),
                        p_dtr_stage(),
                        p_acol(),
                        p_bcol(),
                        p_ccol(),
                        p_dcol()
                ) // esf_descriptor
            ),
            gridtools::make_mss
            (
                execute<backward>(),
                gridtools::make_esf<u_backward_function<double> >(
                        p_utens_stage(),
                        p_u_pos(),
                        p_dtr_stage(),
                        p_ccol(),
                        p_dcol(),
                        p_data_col()
                )
            ),
            domain,
            coords
        );

    vertical_advection->ready();

    vertical_advection->steady();
    domain.clone_to_device();

    vertical_advection->run();

    vertical_advection->finalize();

#ifdef CUDA_EXAMPLE
    repository.update_cpu();
#endif

#ifdef BENCHMARK
    std::cout << vertical_advection->print_meter() << std::endl;
#endif

    verifier verif(1e-10, halo_size);
    bool result = verif.verify(repository.utens_stage_ref(), repository.utens_stage());

    return result;
}

}//namespace vertical_advection
