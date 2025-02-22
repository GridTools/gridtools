ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY . /gridtools

ARG BUILD_TYPE

ENV GTRUN_BUILD_COMMAND='make -j 32'
ENV GTCMAKE_Boost_NO_BOOST_CMAKE=ON
ENV GTCMAKE_Boost_NO_SYSTEM_PATHS=ON
ENV GTCMAKE_GT_TESTS_REQUIRE_FORTRAN_COMPILER=ON
ENV GTCMAKE_GT_TESTS_REQUIRE_C_COMPILER=ON
ENV GTCMAKE_GT_TESTS_REQUIRE_OpenMP=ON
ENV GTCMAKE_GT_TESTS_REQUIRE_GPU=ON
ENV GTCMAKE_GT_TESTS_REQUIRE_Python=ON
ENV GT_ENABLE_STENCIL_DUMP=ON
ENV GTCMAKE_CMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON

# move to base image
ENV CXX=/usr/local/mpich/bin/mpicxx
ENV CC=/usr/local/mpich/bin/mpicc

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN uv run /gridtools/pyutils/driver.py -v build -b ${BUILD_TYPE} -o build -i install -t install || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }
