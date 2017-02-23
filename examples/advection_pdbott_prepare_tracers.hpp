#pragma once

#include <stencil-composition/stencil-composition.hpp>
#include "benchmarker.hpp"

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

namespace adv_prepare_tracers {

    using namespace gridtools;
    using namespace enumtype;
    using namespace expressions;

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > interval_t;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    struct prepare_tracers {
        using data = vector_accessor< 0, inout >;
        using data_nnow = vector_accessor< 1, in >;
        using rho = accessor< 2, in >;
        typedef boost::mpl::vector< data, data_nnow, rho > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, interval_t) {
            eval(data()) = eval(rho()) * eval(data_nnow());
        }
    };

    bool test(uint_t d1, uint_t d2, uint_t d3, uint_t t_steps) {

        typedef BACKEND::storage_traits_t::storage_info_t< 23, 3 > meta_data_t;
        typedef BACKEND::storage_traits_t::data_store_t< float_type, meta_data_t > storage_t;

        meta_data_t meta_data_(d1, d2, d3);

        std::vector< storage_t > list_out_;
        std::vector< storage_t > list_in_;

        for(unsigned i=0; i<20; ++i) {
            list_out_.push_back(storage_t(meta_data_, 0.0));
            list_in_.push_back(storage_t(meta_data_, i));
        }

        storage_t rho(meta_data_, 1.1);

        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis > grid_(di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

        typedef arg< 0, std::vector< storage_t > > p_list_out;
        typedef arg< 1, std::vector< storage_t > > p_list_in;
        typedef arg< 2, storage_t > p_rho;
        typedef boost::mpl::vector< p_list_out, p_list_in, p_rho > args_t;

        aggregator_type< args_t > domain_(list_out_, list_in_, rho);
        auto comp_ =
            make_computation< BACKEND >(expand_factor< 3 >(),
                domain_,
                grid_,
                make_multistage(enumtype::execute< enumtype::forward >(),
                    make_stage< prepare_tracers >(p_list_out(), p_list_in(), p_rho())));

        comp_->ready();
        comp_->steady();
        comp_->run();

#ifdef BENCHMARK
        benchmarker::run(comp_, t_steps);
#endif
        comp_->finalize();

        bool result = true;
        for(unsigned i=0; i<20; ++i) {
            auto out_v = make_host_view(list_out_[i]);
            auto in_v = make_host_view(list_in_[i]);
            auto rho_v = make_host_view(rho);
            for(unsigned a=0; a<d1; ++a) {
                for(unsigned b=0; b<d2; ++b) {
                    for(unsigned c=0; c<d3; ++c) {
                        if(out_v(a,b,c) != rho_v(a,b,c)*in_v(a,b,c)) {
                            std::cout << "error in out field " << i << " in " << a << " " << b << " " << c << ": " << out_v(a,b,c) << " == " << rho_v(a,b,c) << std::endl;
                            result = false;
                        }
                    }
                }
            }
        }

        return result;
    }
}
