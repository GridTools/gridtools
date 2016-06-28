#pragma once
#define FUSION_MAX_VECTOR_SIZE 40
#define FUSION_MAX_MAP_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_LIMIT_VECTOR_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS

#include <stencil-composition/stencil-composition.hpp>
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

    bool test(uint_t d1, uint_t d2, uint_t d3, uint_t t) {

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
    }
} // namespace test_expandable_parameters
