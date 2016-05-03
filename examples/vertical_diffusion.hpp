#pragma once
#include <stencil-composition/stencil-composition.hpp>

namespace vertical_diffusion{

    using namespace gridtools;
    using namespace enumtype;
    using namespace expressions;

    functor vertical_diffusion_stages{


        using data=accessor<0, in>;
        using datatens=accessor<1, in>;
        using data_nnow=accessor<2, in>;
        using sqrtgrhors=accessor<3, in>;
        using zdtr=accessor<4, in>;
        using a1t=accessor<5, in>;
        using a2t=accessor<6, in>;
        using a1tsurf=accessor<7, in>;
        using a2tsurf=accessor<8, in>;
        using kh=accessor<9, in>;
        using acol=accessor<10, in>;
        using bcol=accessor<11, in>;
        using ccol=accessor<12, in>;
        using dcol=accessor<13, in>;

        using data_s=accessor<14, in>;
        using vdtch=accessor<15, in>;
        using bottomFactor=accessor<16, in>;

        dimension<0>::index i;
        dimension<1>::index j;
        dimension<2>::index k;

        template <typename Offset, typename P>
            static constexpr delta(P) {return P(k-1)-P();}

        template <typename Evaluation>
            GT_FUNCTION
            static void Do(Evaluation eval, interval_t)
        {

            T gat = -ctx[kh::Center()] * ctx[sqrtgrhors::Center()];
            T gct = -ctx[vdtch::Center()] * ctx[sqrtgrhors::Center()];
            T as = gat * ctx[a2t::Center()];
            T cs = gct * ctx[a2tsurf::Center()];

            T acolCenter = gat * ctx[a1t::Center()];
            T ccolCenter = gct * ctx[a1tsurf::Center()];;
            //                            acolCenter              ccolCenter
            eval(bcol()) = eval(zdtr() +kh()*sqrtgrhors()*a1t() + vdtch()*sqrtgrhors()*a1tsurf());
            eval(acol()) = -kh()*sqrtgrhors()*a1t(); //acolCenter;
            eval(ccol()) = -vdtch()*sqrtgrhors()*a1tsurf(); //ccolCenter;

            eval(dcol()) = eval(
                data() * zdtr() +
                datatens() +
                kh()*sqrtgrhors()*a2t()// as
                * delta(data_nnow()) -
                vdtch()*sqrtgrhors()*a2tsurf() /*cs*/
                * bottomFactor() * data_nnow() +
                vdtch()*sqrtgrhors()*data_s());

            eval(call<tridiagonal::forward, interval_t>::at<0,0,0>::with(eval, acol(), bcol(), ccol(), dcol()));
        }

        template <typename Evaluation>
            GT_FUNCTION
            static void Do(Evaluation eval, interval_t)
        {
            T gat = -ctx[kh::Center()] * ctx[sqrtgrhors::Center()];
            T gct = -ctx[kh::At(kplus1)] * ctx[sqrtgrhors::Center()];
            T as = gat * ctx[a2t::Center()];
            T cs = gct * ctx[a2t::At(kplus1)];

            T acolCenter = gat * ctx[a1t::Center()];
            T ccolCenter = gct * ctx[a1t::At(kplus1)];
            eval(bcol()) = eval(zdtr() + kh()*sqrtgrhors()*a1t()// acolCenter
                                + kh(k+1)*sqrtgrhors()*at2(k+1));
            eval(acol()) = eval(zdtr() + kh()*sqrtgrhors()*a1t());
            eval(Center()) = eval(kh(k+1)*sqrtgrhors()*at2(k+1));

            eval(dcol()) =
                eval(data() * zdtr() +
                     datatens() +
                     kh()*sqrtgrhors()*a2t()// as
                     * delta(data_nnow()) +
                     kh(k+1)*sqrtgrhors()*a2t(k+1)// cs
                     * delta(data_nnow()));

            call<tridiagonal::forward, interval_t>::at<0,0,0>::with(eval, acol(), bcol(), ccol(), dcol());
        }

        template <typename Evaluation>
            GT_FNUCTION
            static void Do(Evaluation eval, interval_t)
        {
            T gct = -ctx[kh::At(kplus1)] * ctx[sqrtgrhors::Center()];
            T cs = gct * ctx[a2t::At(kplus1)];

            T ccolCenter = gct * ctx[a1t::At(kplus1)];
            eval(bcol()) = eval(zdtr() + kh(k+1)*sqrtgrhors()*a2t(k+1)*a2t(k+1));//ccolCenter
            eval(ccol()) = ccolCenter;

            eval(dcol()) =
                eval(data() * zdtr() +
                     datatens() +
                     kh(k+1)*sqrtgrhors()*a2t(k+1)//cs
                     * delta(data_nnow()));

            call<tridiagonal::forward, interval_t>::at<0,0,0>::with(eval, acol(), bcol(), ccol(), dcol())];
    }

    int main(){


#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
    typedef gridtools::layout_map< 2, 1, 0 > layout_t; // stride 1 on i
#else
    //                   strides   1  x  xy
    //                      dims   x  y  z
    typedef gridtools::layout_map< 0, 1, 2 > layout_t; // stride 1 on k
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_data_t;
        typedef BACKEND::storage_type< float_type, meta_data_t >::type storage_data_t;

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_datatens_t;
        typedef BACKEND::storage_type< float_type, meta_datatens_t >::type storage_datatens_t;

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_data_nnow_t;
        typedef BACKEND::storage_type< float_type, meta_data_nnow_t >::type storage_data_nnow_t;

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_sqrtgrhors_t;
        typedef BACKEND::storage_type< float_type, meta_sqrtgrhors_t >::type storage_sqrtgrhors_t;

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_zdtr_t;
        typedef BACKEND::storage_type< float_type, meta_zdtr_t >::type storage_zdtr_t;

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_a1t_t;
        typedef BACKEND::storage_type< float_type, meta_a1t_t >::type storage_a1t_t;

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_a2t_t;
        typedef BACKEND::storage_type< float_type, meta_a2t_t >::type storage_a2t_t;

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_a1tsurf_t;
        typedef BACKEND::storage_type< float_type, meta_a1tsurf_t >::type storage_a1tsurf_t;

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_a2tsurf_t;
        typedef BACKEND::storage_type< float_type, meta_a2tsurf_t >::type storage_a2tsurf_t;

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_kh_t;
        typedef BACKEND::storage_type< float_type, meta_kh_t >::type storage_kh_t;

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_acol_t;
        typedef BACKEND::storage_type< float_type, meta_acol_t >::type storage_acol_t;

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_bcol_t;
        typedef BACKEND::storage_type< float_type, meta_bcol_t >::type storage_bcol_t;

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_ccol_t;
        typedef BACKEND::storage_type< float_type, meta_ccol_t >::type storage_ccol_t;

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_dcol_t;
        typedef BACKEND::storage_type< float_type, meta_dcol_t >::type storage_dcol_t;

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_data_s_t;
        typedef BACKEND::storage_type< float_type, meta_data_s_t >::type storage_data_s_t;

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_vdtch_t;
        typedef BACKEND::storage_type< float_type, meta_vdtch_t >::type storage_vdtch_t;

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_bottomFactor_t;
        typedef BACKEND::storage_type< float_type, meta_bottomFactor_t >::type storage_bottomFactor_t;


        meta_data_t meta_data_(d1,d2,d3);
        meta_datatens_t meta_datatens_(d1,d2,d3);
        meta_data_nnow_t meta_data_nnow_(d1,d2,d3);
        meta_sqrtgrhors_t meta_sqrtgrhors_(d1,d2,d3);
        meta_zdtr_t meta_zdtr_(d1,d2,d3);
        meta_a1t_t meta_a1t_(d1,d2,d3);
        meta_a2t_t meta_a2t_(d1,d2,d3);
        meta_a1tsurf_t meta_a1tsurf_(d1,d2,d3);
        meta_a2tsurf_t meta_a2tsurf_(d1,d2,d3);
        meta_kh_t meta_kh_(d1,d2,d3);
        meta_acol_t meta_acol_(d1,d2,d3);
        meta_bcol_t meta_bcol_(d1,d2,d3);
        meta_ccol_t meta_ccol_(d1,d2,d3);
        meta_dcol_t meta_dcol_(d1,d2,d3);
        meta_data_s_t meta_data_s_(d1,d2,d3);
        meta_vdtch_t meta_vdtch_(d1,d2,d3);
        meta_bottomFactor_t meta_bottomFactor_(d1,d2,d3);

        storage_data_t storage_data0_(meta_data_);
        storage_data_t storage_data1_(meta_data_);
        storage_data_t storage_data2_(meta_data_);
        storage_data_t storage_data3_(meta_data_);
        storage_data_t storage_data4_(meta_data_);
        storage_data_t storage_data5_(meta_data_);
        storage_data_t storage_data6_(meta_data_);
        storage_data_t storage_data7_(meta_data_);
        storage_data_t storage_data8_(meta_data_);
        storage_data_t storage_data9_(meta_data_);
        storage_data_t storage_data10_(meta_data_);
        storage_data_t storage_data11_(meta_data_);
        storage_data_t storage_data12_(meta_data_);
        storage_data_t storage_data13_(meta_data_);
        storage_data_t storage_data14_(meta_data_);
        storage_data_t storage_data15_(meta_data_);
        storage_data_t storage_data16_(meta_data_);
        storage_data_t storage_data17_(meta_data_);
        storage_data_t storage_data18_(meta_data_);
        storage_data_t storage_data19_(meta_data_);

        storage_datatens_t storage_datatens_(meta_datatens_);
        storage_data_nnow_t storage_data_nnow_(meta_data_nnow_);
        storage_sqrtgrhors_t storage_sqrtgrhors_(meta_sqrtgrhors_);
        storage_zdtr_t storage_zdtr_(meta_zdtr_);
        storage_a1t_t storage_a1t_(meta_a1t_);
        storage_a2t_t storage_a2t_(meta_a2t_);
        storage_a1tsurf_t storage_a1tsurf_(meta_a1tsurf_);
        storage_a2tsurf_t storage_a2tsurf_(meta_a2tsurf_);
        storage_kh_t storage_kh_(meta_kh_);
        storage_acol_t storage_acol_(meta_acol_);
        storage_bcol_t storage_bcol_(meta_bcol_);
        storage_ccol_t storage_ccol_(meta_ccol_);
        storage_dcol_t storage_dcol_(meta_dcol_);
        storage_data_s_t storage_data_s_(meta_data_s_);
        storage_vdtch_t storage_vdtch_(meta_vdtch_);
        storage_bottomFactor_t storage_bottomFactor_(meta_bottomFactor_);

        std::vector<storage_data_t> storage_data_={storage_data0_, storag_data1_, storage_data2_, storage_data3_, storage_data4_, storage_data5_, storage_data6_, storage_data7_, storage_data8_, storage_data9_, storage_data10_, storage_data11_, storage_data12_, storage_data13_, storage_data14_, storage_data15_, storage_data16_, storage_data17_, storage_data18_, storage_data19_};

        typedef arg<0 , std::vector<storage_data_t> > p_data;
        typedef arg<1 , storage_datatens_t> p_datatens;
        typedef arg<2 , storage_data_nnow_t> p_data_nnow;
        typedef arg<3 , storage_sqrtgrhors_t> p_sqrtgrhors;
        typedef arg<4 , storage_zdtr_t> p_zdtr;
        typedef arg<5 , storage_a1t_t> p_a1t;
        typedef arg<6 , storage_a2t_t> p_a2t;
        typedef arg<7 , storage_a1tsurf_t> p_a1tsurf;
        typedef arg<8 , storage_a2tsurf_t> p_a2tsurf;
        typedef arg<9 , storage_kh_t> p_kh;
        typedef arg<10, storage_acol_t> p_acol;
        typedef arg<11, storage_bcol_t> p_bcol;
        typedef arg<12, storage_ccol_t> p_ccol;
        typedef arg<13, storage_dcol_t> p_dcol;
        typedef arg<14, storage_data_s_t> p_data_s;
        typedef arg<15, storage_vdtch_t> p_vdtch;
        typedef arg<16, storage_bottomFactor_t> p_bottomFactor;

        auto domain_ = make_domain_type(storage_data_, storage_datatens_, storage_data_nnow_, storage_sqrtgrhors_, storage_zdtr_, storage_a1t_, storage_a2t_, storage_a1tsurf_, storage_a2tsurf_, storage_kh_, storage_acol_, storage_bcol_, storage_ccol_, storage_dcol_, storage_data_s_, storage_vdtch_, storage_bottomFactor_);

        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

        auto comp_ = make_computation(
            expand_factor<20>(), domain_, grid_,
            make_mss(
                forward()
                , make_esf<vertical_diffusion_stages>(p_data(), p_datatens(), p_data_nnow(), p_sqrthors(), p_zdtr(), p_a1t(), p_a2t(), p_a1tsurf(), p_a2tsurf(), p_kh(), p_acol(), p_bcol(), p_ccol(), p_data_s(), p_vdtch(), p_storage_bottomFactor())
                )
            );

        comop_->ready();
        comp_->steady();
        comp_->run();
        comp_->finalize();

    }
}
