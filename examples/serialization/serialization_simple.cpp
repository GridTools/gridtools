/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

/*
 * This example tests the internal stencil serialization!
 * For normal usage of Serialbox see: https://github.com/MeteoSwiss-APN/serialbox
 */

#if defined(USE_SERIALBOX)

#include "serialization_setup.hpp"

using namespace gridtools;
using namespace enumtype;

typedef gridtools::interval< level< 0, -1 >, level< 1, 1 > > axis;

struct copy_functor {
    typedef accessor< 0, enumtype::in, extent<>, 3 > in;
    typedef accessor< 1, enumtype::inout, extent<>, 3 > out;

    typedef boost::mpl::vector< in, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        eval(out()) = eval(in());
    }
};

namespace ser = serialbox::gridtools;

class serialization_setup_simple : public serialization_setup {
  private:
  public:
    using p_in = arg< 0, storage_t >;
    using p_out = arg< 1, storage_t >;
    using accessor_list = boost::mpl::vector< p_in, p_out >;

    uint_t d1, d2, d3;
    storage_t in, out;
    gridtools::grid< axis > grid;
    gridtools::aggregator_type< accessor_list > domain;

    decltype(make_computation< backend_t >(
        domain, grid, make_multistage(execute< forward >(), make_stage< copy_functor >(p_in(), p_out())))) copy;

    serialization_setup_simple()
        : d1(3), d2(4), d3(5),                                                               //
          in(make_storage("in", [](int i, int j, int k) { return i + j + k; }, d1, d2, d3)), //
          out(make_storage("out", -1., d1, d2, d3)),                                         //
          grid(halo_descriptor{0, 0, 0, d1 - 1, d1}, halo_descriptor{0, 0, 0, d2 - 1, d2}),  //
          domain((p_in() = in), (p_out() = out)) {
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

        copy = make_computation< backend_t >(
            domain, grid, make_multistage(execute< forward >(), make_stage< copy_functor >(p_in(), p_out())));
    }

    void run_copy_and_serialize() {
        ser::serializer serializer(ser::open_mode::Write, directory(), prefix());
        copy->ready();
        copy->steady();
        copy->run(serializer, "copy");
        copy->finalize();
    }
};

TEST_F(serialization_setup_simple, check_functor) {
    run_copy_and_serialize();
    ASSERT_TRUE(verify_storages(in, out, grid));
}

TEST_F(serialization_setup_simple, check_meta_info) {
    run_copy_and_serialize();
    ser::serializer ref_serializer(ser::open_mode::Read, directory(), prefix());

    ASSERT_TRUE(verify_field_meta_info(ref_serializer));

    ASSERT_EQ(ref_serializer.savepoints().size(), 2);
    ASSERT_EQ(ref_serializer.fieldnames().size(), 2);
    ASSERT_TRUE(ref_serializer.global_meta_info().has_key("stencils"));
    ASSERT_EQ(ref_serializer.get_global_meta_info_as< std::vector< std::string > >("stencils")[0], "copy");
}

TEST_F(serialization_setup_simple, check_serialized_data) {
    run_copy_and_serialize();
    ser::serializer ref_serializer(ser::open_mode::Read, directory(), prefix());

    storage_t copy_input_in = make_storage("copy_input_in", -1., d1, d2, d3);
    storage_t copy_input_out = make_storage("copy_input_out", -1., d1, d2, d3);
    storage_t copy_output_in = make_storage("copy_output_in", -1., d1, d2, d3);
    storage_t copy_output_out = make_storage("copy_output_out", -1., d1, d2, d3);
    ref_serializer.read("in", ref_serializer.savepoints()[0], copy_input_in);
    ref_serializer.read("out", ref_serializer.savepoints()[0], copy_input_out);
    ref_serializer.read("in", ref_serializer.savepoints()[1], copy_output_in);
    ref_serializer.read("out", ref_serializer.savepoints()[1], copy_output_out);

    storage_t copy_input_out_ref = make_storage("copy_input_out_ref", -1., d1, d2, d3);

    ASSERT_TRUE(verify_storages(copy_input_in, in, grid));
    ASSERT_TRUE(verify_storages(copy_input_out, copy_input_out_ref, grid));
    ASSERT_TRUE(verify_storages(copy_output_in, in, grid));
    ASSERT_TRUE(verify_storages(copy_output_out, out, grid));
}

#endif
