#pragma once

#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/stencil-functions/stencil-functions.hpp>
#include "backend_select.hpp"
#include "benchmarker.hpp"
#include <common/gt_math.hpp>

/**
  @file
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend
*/

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

using namespace gridtools;
using namespace gridtools::enumtype;
using namespace gridtools::expressions;

using Real = float_type;

namespace Direction {

    /**
     * @brief Namepsace of the directions
     */

    /**
     * @brief X-direction (i)
     */
    using X = gridtools::dimension< 1 >;

    /**
     * @brief Y-direction (j)
     */
    using Y = gridtools::dimension< 2 >;

    /**
     * @brief Z-direction (k)
     */
    using Z = gridtools::dimension< 3 >;
} // namespace Direction

namespace Index {

    /**
     * @brief Index of i-direction
     */
    constexpr static gridtools::dimension< 1 > i;

    /**
     * @brief Index of j-direction
     */
    constexpr static gridtools::dimension< 2 > j;

    /**
     * @brief Index of k-direction
     */
    constexpr static gridtools::dimension< 3 > k;
} // namespace Index

#define DYCORE_DEFINE_INDICES \
    using namespace Index;    \
    using namespace Direction;

struct Identity {
    using out = inout_accessor< 0 >;
    using in = in_accessor< 1 >;
    using arg_list = boost::mpl::vector< out, in >;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        eval(out()) = eval(in());
    }
};

template < typename Dimension, int distance = 1, typename Function = Identity, typename Eval, typename Accessor >
GT_FUNCTION Real Delta(Eval &eval, Accessor) {
    Real l = call< Function >::with(eval, Accessor(Dimension() + 1 * distance));
    Real r = call< Function >::with(eval, Accessor());
    return l - r;
}

/**
 * @struct FluxStage
 * Compute flux in x and y direction
 * Corresponds to numeric_utilities.f90 - first loop of lap_4aml
 */
struct FluxStage {
    using flx = inout_accessor< 0 >;
    using fly = inout_accessor< 1 >;
    using lap = in_accessor< 2, extent< 0, 1, 0, 1 > >;
    using mask = in_accessor< 3 >;
    using crlato = in_accessor< 4 >;
    using ofahdx = in_accessor< 5 >;
    using ofahdy = in_accessor< 6 >;

    using arg_list = boost::mpl::vector< flx, fly, lap, mask, crlato, ofahdx, ofahdy >;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        DYCORE_DEFINE_INDICES
        computeFluxes(eval, eval(mask() * ofahdx()), eval(mask() * ofahdy()));
    }

  private:
    template < typename Evaluation >
    GT_FUNCTION static void computeFluxes(Evaluation &eval, const Real maskX, const Real maskY) {
        DYCORE_DEFINE_INDICES
        eval(flx()) = maskX * Delta< X, 1 >(eval, lap());
        eval(fly()) = maskY * Delta< Y, 1 >(eval, lap()) * eval(crlato());
    }
};

/**
 * @struct RXStage
 * Corresponds to numeric_utilities.f90 - second loop of lap_4aml
 */
struct RXStage {
    using rxp = inout_accessor< 0 >;
    using rxm = inout_accessor< 1 >;
    using flx = in_accessor< 2, extent< -1, 0, 0, 0 > >;
    using fly = in_accessor< 3, extent< 0, 0, -1, 0 > >;
    using s = in_accessor< 4, extent< -1, 1, -1, 1 > >;

    using arg_list = boost::mpl::vector< rxp, rxm, flx, fly, s >;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        DYCORE_DEFINE_INDICES

        Real fmax = math::max(eval(s()), eval(s(i - 1)), eval(s(i + 1)), eval(s(j - 1)), eval(s(j + 1)));
        Real fmin = math::min(eval(s()), eval(s(i - 1)), eval(s(i + 1)), eval(s(j - 1)), eval(s(j + 1)));

        Real fluxin = computeFlux(eval(flx(i - 1)), eval(flx())) + computeFlux(eval(fly(j - 1)), eval(fly()));
        Real fluxou = computeFlux(eval(flx()), eval(flx(i - 1))) + computeFlux(eval(fly()), eval(fly(j - 1)));

        eval(rxp()) = math::fabs(fmax - eval(s())) / (fluxin + (Real)1e-35);
        eval(rxm()) = math::fabs(fmin - eval(s())) / (fluxou + (Real)1e-35);
    }

  private:
    GT_FUNCTION static Real computeFlux(const Real from, const Real to) {
        return math::max((Real)0.0, from) - math::min((Real)0.0, to);
    }
};

/**
 * @struct LimitFluxStage
 * Limit the flux in x and y direction
 * Corresponds to numeric_utilities.f90 - third loop of lap_4aml
 */
struct LimitFluxStage {
    using flx = inout_accessor< 0 >;
    using fly = inout_accessor< 1 >;
    using rxp = in_accessor< 2, extent< 0, 1, 0, 1 > >;
    using rxm = in_accessor< 3, extent< 0, 1, 0, 1 > >;

    using arg_list = boost::mpl::vector< flx, fly, rxp, rxm >;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        DYCORE_DEFINE_INDICES
        eval(flx()) = limiter< X >(eval, flx());
        eval(fly()) = limiter< Y >(eval, fly());
    }

    template < typename Offset, typename Eval, typename Accessor >
    GT_FUNCTION static Real limiter(Eval &eval, Accessor) {
        Real flux = eval(Accessor());
        if (flux >= (Real)0.0) {
            flux *= math::min((Real)1.0, eval(rxp(Offset() + 1)), eval(rxm()));
        } else {
            flux *= math::min((Real)1.0, eval(rxp()), eval(rxm(Offset() + 1)));
        }
        return flux;
    }
};

/**
 * @struct DataStage
 * Corresponds to numeric_utilities.f90 - final loop of lap_4aml
 */
struct DataStage {
    using data_out = inout_accessor< 0 >;
    using data_in = in_accessor< 1 >;
    using flx = in_accessor< 2, extent< -1, 0, 0, 0 > >;
    using fly = in_accessor< 3, extent< 0, 0, -1, 0 > >;
    using minimum_value = global_accessor< 4 >;

    using arg_list = boost::mpl::vector< data_out, data_in, flx, fly, minimum_value >;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        DYCORE_DEFINE_INDICES
        eval(data_out()) = math::max(
            eval(minimum_value()), eval(data_in()) + Delta< X, -1 >(eval, flx()) + Delta< Y, -1 >(eval, fly()));
    }
};

struct Laplacian_FullDomain {
    using lap = inout_accessor< 0 >;
    using in = in_accessor< 1, extent< -1, 1, -1, 1 > >;
    using crlato = in_accessor< 2 >;
    using crlatu = in_accessor< 3 >;

    using arg_list = boost::mpl::vector< lap, in, crlato, crlatu >;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        DYCORE_DEFINE_INDICES

        eval(lap()) = eval(
            in(i + 1) + in(i - 1) - (Real)2.0 * in() + crlato() * (in(j + 1) - in()) + crlatu() * (in(j - 1) - in()));
    }
};

namespace hori_diff_type2_limiter {

    bool test(uint_t d1, uint_t d2, uint_t d3) {

        using IJKRealFieldInfo = storage_traits< backend_t::s_backend_id >::storage_info_t< 0, 3, halo< 3, 3, 0 > >;
        using IJKRealField = storage_traits< backend_t::s_backend_id >::data_store_t< float_type, IJKRealFieldInfo >;

        using JRealFieldInfo = storage_traits<
            backend_t::s_backend_id >::special_storage_info_t< 1, gridtools::selector< 0, 1, 0 >, halo< 3, 3, 0 > >;
        using JRealField = storage_traits< backend_t::s_backend_id >::data_store_t< float_type, JRealFieldInfo >;

        IJKRealFieldInfo meta_data_ijk(d1, d2, d3);
        JRealFieldInfo meta_data_j(d1, d2, d3);

        IJKRealField f_data_in(meta_data_ijk);
        IJKRealField f_data_out(meta_data_ijk);
        IJKRealField f_hdmask(meta_data_ijk);
        IJKRealField f_ofahdx(meta_data_ijk);
        IJKRealField f_ofahdy(meta_data_ijk);

        JRealField f_crlato(meta_data_j);
        JRealField f_crlatu(meta_data_j);

        float_type mval = std::numeric_limits< float_type >::lowest();
        auto f_minimum_value = backend_t::make_global_parameter(mval);

        using data_in = gridtools::arg< 0, IJKRealField >;
        using data_out = gridtools::arg< 1, IJKRealField >;
        using hdmask = gridtools::arg< 2, IJKRealField >;
        using ofahdx = gridtools::arg< 3, IJKRealField >;
        using ofahdy = gridtools::arg< 4, IJKRealField >;
        using crlato = gridtools::arg< 5, JRealField >;
        using crlatu = gridtools::arg< 6, JRealField >;

        using minimum_value = gridtools::arg< 7, decltype(f_minimum_value) >;

        using lap = gridtools::tmp_arg< 8, IJKRealField >;
        using flx = gridtools::tmp_arg< 9, IJKRealField >;
        using fly = gridtools::tmp_arg< 10, IJKRealField >;
        using rxp = gridtools::tmp_arg< 11, IJKRealField >;
        using rxm = gridtools::tmp_arg< 12, IJKRealField >;

        halo_descriptor di{3, 3, 3, d1 - 3 - 1, d1};
        halo_descriptor dj{3, 3, 3, d2 - 3 - 1, d2};

        auto grid = make_grid(di, dj, d3);

        auto stencil = make_computation< backend_t >(
            grid,
            (minimum_value() = f_minimum_value),
            make_multistage(
                execute< parallel >(),
                define_caches(cache< IJ, cache_io_policy::local >(rxp())),
                make_stage< Laplacian_FullDomain >(lap(), data_in(), crlato(), crlatu()),
                make_stage< FluxStage >(flx(), fly(), lap(), hdmask(), crlato(), ofahdx(), ofahdy()),
                make_stage< RXStage >(rxp(), rxm(), flx(), fly(), data_in()),
                make_stage< LimitFluxStage >(flx(),
                    fly(),
                    rxp(),
                    rxm()), // TODO: the problem in the dependency analysis is here (LimitFluxStage):it wants
                // to compute extent<-2,1,-2,1>
                make_stage< DataStage >(data_out(), data_in(), flx(), fly(), minimum_value())));

        stencil.run(data_in() = f_data_in,
            data_out() = f_data_out,
            hdmask() = f_hdmask,
            ofahdx() = f_ofahdx,
            ofahdy() = f_ofahdy,
            crlato() = f_crlato,
            crlatu() = f_crlatu);

        return true;
    }
}
