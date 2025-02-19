ARG BASE_IMAGE
ARG BUILD_TYPE=Release
FROM $BASE_IMAGE

COPY . /gridtools

RUN /gridtools/pyutils/driver.py -v build -b $BUILD_TYPE -o build -i install -t install || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }
