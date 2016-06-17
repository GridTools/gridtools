#pragma once
#define FUSION_MAX_VECTOR_SIZE 40
#define FUSION_MAX_MAP_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_LIMIT_VECTOR_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS

#include <stencil_composition/stencil_composition.hpp>
#include <tools/verifier.hpp>

namespace test_expandable_parameters {

    using namespace gridtools;
    using namespace expressions;

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

#ifdef CUDA_EXAMPLE
#define BACKEND backend< enumtype::Cuda, GRIDBACKEND, enumtype::Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< enumtype::Host, GRIDBACKEND, enumtype::Block >
#else
#define BACKEND backend< enumtype::Host, GRIDBACKEND, enumtype::Naive >
#endif
#endif

    struct functor_exp {

#ifdef REASSIGN_DOMAIN
        typedef accessor< 0, enumtype::inout > parameters_out;
        typedef accessor< 1, enumtype::in > parameters_in;
#else
        typedef vector_accessor< 0, enumtype::inout > parameters_out;
        typedef vector_accessor< 1, enumtype::in > parameters_in;
#endif
        // typedef accessor<2, enumtype::in> scalar;

        typedef boost::mpl::vector< parameters_out, parameters_in //, scalar
            > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {

            eval(parameters_out{}) = eval(parameters_in{});
        }
    };

    struct functor_single_kernel {

        typedef accessor< 0, enumtype::inout > parameters1_out;
        typedef accessor< 1, enumtype::inout > parameters2_out;
        typedef accessor< 2, enumtype::inout > parameters3_out;
        typedef accessor< 3, enumtype::inout > parameters4_out;
        typedef accessor< 4, enumtype::inout > parameters5_out;
        typedef accessor< 5, enumtype::inout > parameters6_out;
        typedef accessor< 6, enumtype::inout > parameters7_out;
        typedef accessor< 7, enumtype::inout > parameters8_out;

        typedef accessor< 8, enumtype::in > parameters1_in;
        typedef accessor< 9, enumtype::in > parameters2_in;
        typedef accessor< 10, enumtype::in > parameters3_in;
        typedef accessor< 11, enumtype::in > parameters4_in;
        typedef accessor< 12, enumtype::in > parameters5_in;
        typedef accessor< 13, enumtype::in > parameters6_in;
        typedef accessor< 14, enumtype::in > parameters7_in;
        typedef accessor< 15, enumtype::in > parameters8_in;
        // typedef accessor<2, enumtype::in> scalar;

        typedef boost::mpl::vector< parameters1_out,
            parameters2_out,
            parameters3_out,
            parameters4_out,
            parameters5_out,
            parameters6_out,
            parameters7_out,
            parameters8_out,
            parameters1_in,
            parameters2_in,
            parameters3_in,
            parameters4_in,
            parameters5_in,
            parameters6_in,
            parameters7_in,
            parameters8_in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(parameters1_out()) = eval(parameters1_in());
            eval(parameters2_out()) = eval(parameters2_in());
            eval(parameters3_out()) = eval(parameters3_in());
            eval(parameters4_out()) = eval(parameters4_in());
            eval(parameters5_out()) = eval(parameters5_in());
            eval(parameters6_out()) = eval(parameters6_in());
            eval(parameters7_out()) = eval(parameters7_in());
            eval(parameters8_out()) = eval(parameters8_in());
        }
    };

    bool test(uint_t d1, uint_t d2, uint_t d3, uint_t t) {

#ifdef CUDA8

#ifdef CUDA_EXAMPLE
        typedef layout_map< 2, 1, 0 > layout_t;
#else
        typedef layout_map< 0, 1, 2 > layout_t;
#endif

        typedef BACKEND::storage_info< 23, layout_t > meta_data_t;
        typedef BACKEND::storage_type< float_type, meta_data_t >::type storage_t;
        typedef BACKEND::temporary_storage_type< float_type, meta_data_t >::type tmp_storage_t;

        typedef storage_t storage_t;

        meta_data_t meta_data_(d1, d2, d3);

        storage_t storage1(meta_data_, 1., "storage1");
        storage_t storage2(meta_data_, 2., "storage2");
        storage_t storage3(meta_data_, 3., "storage3");
        storage_t storage4(meta_data_, 4., "storage4");
        storage_t storage5(meta_data_, 5., "storage5");
        storage_t storage6(meta_data_, 6., "storage6");
        storage_t storage7(meta_data_, 7., "storage7");
        storage_t storage8(meta_data_, 8., "storage8");

        storage_t storage10(meta_data_, -1., "storage10");
        storage_t storage20(meta_data_, -2., "storage20");
        storage_t storage30(meta_data_, -3., "storage30");
        storage_t storage40(meta_data_, -4., "storage40");
        storage_t storage50(meta_data_, -5., "storage50");
        storage_t storage60(meta_data_, -6., "storage60");
        storage_t storage70(meta_data_, -7., "storage70");
        storage_t storage80(meta_data_, -8., "storage80");

        std::vector< pointer< storage_t > > list_out_ = {
            &storage1, &storage2, &storage3, &storage4, &storage5, &storage6, &storage7, &storage8};
        std::vector< pointer< storage_t > > list_in_ = {
            &storage10, &storage20, &storage30, &storage40, &storage50, &storage60, &storage70, &storage80};

        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis > grid_(di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

#ifdef SINGLE_KERNEL

        typedef arg< 0, storage_t > p_0_out;
        typedef arg< 1, storage_t > p_1_out;
        typedef arg< 2, storage_t > p_2_out;
        typedef arg< 3, storage_t > p_3_out;
        typedef arg< 4, storage_t > p_4_out;
        typedef arg< 5, storage_t > p_5_out;
        typedef arg< 6, storage_t > p_6_out;
        typedef arg< 7, storage_t > p_7_out;

        typedef arg< 8, storage_t > p_0_in;
        typedef arg< 9, storage_t > p_1_in;
        typedef arg< 10, storage_t > p_2_in;
        typedef arg< 11, storage_t > p_3_in;
        typedef arg< 12, storage_t > p_4_in;
        typedef arg< 13, storage_t > p_5_in;
        typedef arg< 14, storage_t > p_6_in;
        typedef arg< 15, storage_t > p_7_in;

        typedef arg< 16, tmp_storage_t > p_0_tmp;
        typedef arg< 17, tmp_storage_t > p_1_tmp;
        typedef arg< 18, tmp_storage_t > p_2_tmp;
        typedef arg< 19, tmp_storage_t > p_3_tmp;
        typedef arg< 20, tmp_storage_t > p_4_tmp;
        typedef arg< 21, tmp_storage_t > p_5_tmp;
        typedef arg< 22, tmp_storage_t > p_6_tmp;
        typedef arg< 23, tmp_storage_t > p_7_tmp;

        typedef boost::mpl::vector< p_0_out,
            p_1_out,
            p_2_out,
            p_3_out,
            p_4_out,
            p_5_out,
            p_6_out,
            p_7_out,
            p_0_in,
            p_1_in,
            p_2_in,
            p_3_in,
            p_4_in,
            p_5_in,
            p_6_in,
            p_7_in,
            p_0_tmp,
            p_1_tmp,
            p_2_tmp,
            p_3_tmp,
            p_4_tmp,
            p_5_tmp,
            p_6_tmp,
            p_7_tmp > args_t;

        aggregator_type< args_t > domain_(boost::fusion::make_vector(&storage1,
            &storage2,
            &storage3,
            &storage4,
            &storage5,
            &storage6,
            &storage7,
            &storage8,
            &storage10,
            &storage20,
            &storage30,
            &storage40,
            &storage50,
            &storage60,
            &storage70,
            &storage80));

        auto comp_ = make_computation< BACKEND >(
            domain_,
            grid_,
            make_multistage(enumtype::execute< enumtype::forward >(),
                define_caches(cache< IJ, local >(
                    p_0_tmp(), p_1_tmp(), p_2_tmp(), p_3_tmp(), p_4_tmp(), p_5_tmp(), p_6_tmp(), p_7_tmp())),
                make_stage< functor_single_kernel >(p_0_tmp(),
                         p_1_tmp(),
                         p_2_tmp(),
                         p_3_tmp(),
                         p_4_tmp(),
                         p_5_tmp(),
                         p_6_tmp(),
                         p_7_tmp(),
                         p_0_in(),
                         p_1_in(),
                         p_2_in(),
                         p_3_in(),
                         p_4_in(),
                         p_5_in(),
                         p_6_in(),
                         p_7_in()),
                make_stage< functor_single_kernel >(p_0_out(),
                         p_1_out(),
                         p_2_out(),
                         p_3_out(),
                         p_4_out(),
                         p_5_out(),
                         p_6_out(),
                         p_7_out(),
                         p_0_tmp(),
                         p_1_tmp(),
                         p_2_tmp(),
                         p_3_tmp(),
                         p_4_tmp(),
                         p_5_tmp(),
                         p_6_tmp(),
                         p_7_tmp())));

        comp_->ready();
        comp_->steady();
        comp_->run();
        comp_->finalize();
#else
#ifdef REASSIGN_DOMAIN

        typedef arg< 0, storage_t > p_out;
        typedef arg< 1, storage_t > p_in;
        typedef arg< 2, tmp_storage_t > p_tmp;

        typedef boost::mpl::vector< p_out, p_in, p_tmp > args_t;

        aggregator_type< args_t > domain_((p_out() = *list_out_[0]), (p_in() = *list_in_[0]));

        auto comp_ = make_computation< BACKEND >(domain_,
            grid_,
            make_multistage(enumtype::execute< enumtype::forward >(),
                                                     define_caches(cache< IJ, local >(p_tmp())),
                                                     make_stage< functor_exp >(p_tmp(), p_in()),
                                                     make_stage< functor_exp >(p_out(), p_tmp())));

        for (uint_t i = 0; i < list_in_.size(); ++i) {
            comp_->ready();
            comp_->reassign((p_in() = *list_in_[i]), (p_out() = *list_out_[i]));
            comp_->steady();
            comp_->run();
            finalize_computation< BACKEND::s_backend_id >::apply(domain_);
        }
        comp_->finalize();

#else // REASSIGN_DOMAIN

        typedef arg< 0, std::vector< pointer< storage_t > > > p_list_out;
        typedef arg< 1, std::vector< pointer< storage_t > > > p_list_in;
        typedef arg< 2, std::vector< pointer< tmp_storage_t > > > p_list_tmp;

        typedef boost::mpl::vector< p_list_out, p_list_in, p_list_tmp > args_t;

        aggregator_type< args_t > domain_(boost::fusion::make_vector(&list_out_, &list_in_));

        auto comp_ = make_computation< BACKEND >(expand_factor< 3 >(),
            domain_,
            grid_,
            make_multistage(enumtype::execute< enumtype::forward >(),
                            define_caches(cache< IJ, local >(p_list_tmp())),
                                                     make_stage< functor_exp >(p_list_tmp(), p_list_in()),
                                                     make_stage< functor_exp >(p_list_out(), p_list_tmp())));

        comp_->ready();
        comp_->steady();
        comp_->run();
        comp_->finalize();

#endif // REASSIGN_DOMAIN
#endif // SINGLE_KERNEL

        bool success = true;
        for (uint_t l = 0; l < list_in_.size(); ++l)
            for (uint_t i = 0; i < d1; ++i)
                for (uint_t j = 0; j < d2; ++j)
                    for (uint_t k = 0; k < d3; ++k) {
                        if ((*list_in_[l])(i, j, k) != (*list_out_[l])(i, j, k)) {
                            std::cout << "error in " << i << ", " << j << ", " << k << ": "
                                      << "in = " << (*list_in_[l])(i, j, k) << ", out = " << (*list_out_[l])(i, j, k)
                                      << std::endl;
                            success = false;
                        }
                    }

        return success;
#else // CUDA8
        return true;
#endif // CUDA8
    }
} // namespace test_expandable_parameters
