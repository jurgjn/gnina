#!/bin/bash
# Build all C++ dependencies for the gnina-classic fork into deps/prefix
set -e
cd /cluster/scratch/jjaenes/in-vina-veritas/deps
SRC=src
PFX=prefix
JOBS=32

echo "=== zlib ==="
tar xzf $SRC/zlib-1.3.1.tar.gz
(cd $SRC/zlib-1.3.1 && ./configure --prefix=$PWD/../$PFX && make -s -j$JOBS && make -s install)

echo "=== boost (subset) ==="
tar xzf $SRC/boost_1_84_0.tar.gz
cd $SRC/boost_1.84.0
./bootstrap.sh --prefix=$PWD/../$PFX --with-libraries=program_options,iostreams,timer,thread,serialization,filesystem,date_time,regex,system 2>&1 | tail -2
./b2 -j$JOBS --without-python cxxstd=17 variant=release link=static 2>&1 | tail -3
cd -

echo "=== jsoncpp ==="
tar xzf $SRC/jsoncpp-1.9.5.tar.gz
cmake -S $SRC/jsoncpp-1.9.5 -B $SRC/jsoncpp-build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PFX -DJSONCPP_BUILD_TESTS=OFF -DJSONCPP_BUILD_APPS=ON > /dev/null
cmake --build $SRC/jsoncpp-build -j$JOBS > /dev/null && cmake --install $SRC/jsoncpp-build > /dev/null

echo "=== openbabel ==="
cd $SRC/openbabel-3.1.1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PFX \
  -DBUILD_CPP_TESTS=OFF -DBUILD_PYTHON=OFF -DBUILD_TESTS=OFF \
  -DZLIB_ROOT=$PFX -DCMAKE_PREFIX_PATH=$PFX > /dev/null
cmake --build build -j$JOBS 2>&1 | tail -2
cmake --install build > /dev/null

echo "ALL DEPS BUILT"
ls $PFX/lib | head -20
