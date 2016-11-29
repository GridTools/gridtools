/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
 * For normal usage of Serialbox see: https://thfabian.github.io/serialbox2
 */

#if defined(USE_SERIALBOX) && defined(CXX11_ENABLED)

#include "serialization_setup.hpp"

using namespace gridtools;
using namespace enumtype;

typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

struct copy_functor {
    typedef accessor< 0, enumtype::in, extent<>, 3 > in;
    typedef accessor< 1, enumtype::inout, extent<>, 3 > out;

    typedef boost::mpl::vector< in, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
        eval(out()) = eval(in());
    }
};

TEST_F(serialization_setup, simple) {
    namespace ser = serialbox::gridtools;
    uint_t d1 = 3, d2 = 4, d3 = 5;

    // Storages
    storage_t &in = make_storage("in", d1, d2, d3);
    storage_t &out = make_storage("out", d1, d2, d3);

    for_each("in", [&in](int i, int j, int k) { in(i, j, k) = i + j + k; });
    for_each("out", [&out](int i, int j, int k) { out(i, j, k) = -1; });

    // Domain
    typedef arg< 0, storage_t > p_in;
    typedef arg< 1, storage_t > p_out;
    typedef boost::mpl::vector< p_in, p_out > accessor_list;

    gridtools::aggregator_type< accessor_list > domain((p_in() = in), (p_out() = out));

    // Grid
    uint_t di[5] = {0, 0, 0, d1 - 1, d1};
    uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

    gridtools::grid< axis > grid(di, dj);
    grid.value_list[0] = 0;
    grid.value_list[1] = d3 - 1;

    // Assemble stencil
    auto copy = make_computation< backend_t >(
        domain, grid, make_multistage(execute< forward >(), make_stage< copy_functor >(p_in(), p_out())));

    try {
        // Run & serialize
        // ===============
        ser::serializer serializer(ser::open_mode::Write, directory(), prefix());
        copy->ready();
        copy->steady();
        copy->run(serializer, "copy");
        copy->finalize();

        // Verify
        // ======
        ser::serializer ref_serializer(ser::open_mode::Read, directory(), prefix());

        // Check functor
        ASSERT_TRUE(verify_storages(in, out, grid));

        // Check field meta-information
        ASSERT_TRUE(verify_field_meta_info(ref_serializer));

        // Check serialized data
        ASSERT_EQ(ref_serializer.savepoints().size(), 2);
        ASSERT_EQ(ref_serializer.fieldnames().size(), 2);
        ASSERT_TRUE(ref_serializer.global_meta_info().has_key("stencils"));
        ASSERT_EQ(ref_serializer.get_global_meta_info_as< std::vector< std::string > >("stencils")[0], "copy");

        // Load serialized data
        // ====================
        storage_t &copy_input_in = make_storage("copy_input_in", d1, d2, d3);
        storage_t &copy_input_out = make_storage("copy_input_out", d1, d2, d3);
        storage_t &copy_output_in = make_storage("copy_output_in", d1, d2, d3);
        storage_t &copy_output_out = make_storage("copy_output_out", d1, d2, d3);

        ref_serializer.read("in", ref_serializer.savepoints()[0], copy_input_in);
        ref_serializer.read("out", ref_serializer.savepoints()[0], copy_input_out);
        ref_serializer.read("in", ref_serializer.savepoints()[1], copy_output_in);
        ref_serializer.read("out", ref_serializer.savepoints()[1], copy_output_out);

        // Verify serialized data
        // ======================
        storage_t &copy_input_out_ref = make_storage("copy_input_out_ref", d1, d2, d3);
        for_each(
            "copy_input_out_ref", [&copy_input_out_ref](int i, int j, int k) { copy_input_out_ref(i, j, k) = -1; });

        ASSERT_TRUE(verify_storages(copy_input_in, in, grid));
        ASSERT_TRUE(verify_storages(copy_input_out, copy_input_out_ref, grid));
        ASSERT_TRUE(verify_storages(copy_output_in, in, grid));
        ASSERT_TRUE(verify_storages(copy_output_out, out, grid));

    } catch (std::exception &e) {
        ASSERT_TRUE(false) << e.what();
    }
}

#endif
