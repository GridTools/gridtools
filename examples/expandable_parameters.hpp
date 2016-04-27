#pragma once
#include <stencil-composition/stencil-composition.hpp>

namespace test_expandable_parameters{

    using namespace gridtools;
    using namespace expressions;

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

#ifdef CUDA_EXAMPLE
#define BACKEND backend< enumtype::Cuda, enumtype::Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< enumtype::Host, enumtype::Block >
#else
#define BACKEND backend< enumtype::Host, enumtype::Naive >
#endif
#endif

    struct functor{


        typedef vector_accessor<0, enumtype::inout> parameters;
        // typedef vector_accessor<1, enumtype::in>    parameters2;
        // typedef accessor<2, enumtype::in> scalar;

        typedef boost::mpl::vector<parameters// , parameters2, scalar
                                   > arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const& eval, x_interval){

            printf("value: %f\n",eval(parameters(0,0,0,0)));
            //eval(parameters(0,0,0))=0.;//eval(parameters2(1,0,0)+scalar(0,0,0));
        }
    };

    bool test(uint_t d1, uint_t d2, uint_t d3, uint_t t){

#define N_PARAMS 23

#ifdef CUDA_EXAMPLE
        typedef layout_map<2,1,0> layout_t;
#else
        typedef layout_map<0,1,2> layout_t;
#endif

        typedef BACKEND::storage_info< 23, layout_t > meta_data_t;
        typedef BACKEND::storage_type< float_type, meta_data_t >::type storage_t;

        typedef storage_t storage_t;

        meta_data_t meta_data_(d1, d2, d3);

        typedef arg<0, std::vector<pointer<storage_t> > > p_list;

        storage_t storage1(meta_data_, 0., "storage1");
        storage_t storage2(meta_data_, 1., "storage2");
        storage_t storage3(meta_data_, 2., "storage3");
        storage_t storage4(meta_data_, 3., "storage4");
        storage_t storage5(meta_data_, 4., "storage5");
        storage_t storage6(meta_data_, 5., "storage6");
        storage_t storage7(meta_data_, 6., "storage7");
        storage_t storage8(meta_data_, 7., "storage8");

        std::vector<pointer<storage_t> > list_={&storage1, &storage2, &storage3, &storage4, &storage5, &storage6, &storage7};

        typedef boost::mpl::vector<p_list> args_t;

        domain_type<args_t> domain_(boost::fusion::make_vector(&list_));

        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis > grid_(di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

        auto comp_ = make_computation<BACKEND>(
            expand_factor<4>(), domain_, grid_,
                make_mss(
                    enumtype::execute<enumtype::forward>()
                    , make_esf<functor>(p_list())
                )
            );
        comp_->ready();
        comp_->steady();
        comp_->run();
        comp_->finalize();

        return true;
    }
}// namespace test_expandable_parameters
