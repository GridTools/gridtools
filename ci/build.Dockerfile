ARG BASE_IMAGE
ARG BUILD_TYPE="release"
FROM $BASE_IMAGE

COPY . /gridtools

RUN pip install --user -r /gridtools/pyutils/requirements.txt
RUN echo "{BUILD_TYPE}"
RUN /gridtools/pyutils/driver.py -v build -b $BUILD_TYPE -o build -i install -t install || { echo 'Build failed'; rm -rf $tmpdir; exit 1; }
