ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY . /gridtools

ARG BUILD_TYPE

ENV GTRUN_BUILD_COMMAND='make -j 32'

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN uv run /gridtools/pyutils/driver.py -v build -b ${BUILD_TYPE} -o build -i install -t install || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }
