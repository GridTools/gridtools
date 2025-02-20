ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY . /gridtools

RUN pip install --user -r /gridtools/pyutils/requirements.txt

ARG BUILD_TYPE

ENV GTRUN_BUILD_COMMAND='make -j 32'

RUN /gridtools/pyutils/driver.py -v build -b ${BUILD_TYPE} -o build -i install -t install || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }
