FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY src ./src

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j \
    && install -m 0755 build/benchmark build/correctness_test /usr/local/bin/

WORKDIR /work
ENTRYPOINT ["benchmark"]
