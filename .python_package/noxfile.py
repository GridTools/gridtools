# GridTools
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import shutil

import nox


_ROOT = ".."


nox.options.sessions = ["test_src", "test_wheel"]


@nox.session
def test_src(session: nox.Session):
    session.install(_ROOT)
    session.install("pytest")
    session.run("pytest", "tests", *session.posargs)


@nox.session
def build_wheel(session: nox.Session):
    dist_path = session.cache_dir.joinpath("dist").absolute()
    session.install("build")
    session.run("python", "-m", "build", "--wheel", "-o", str(dist_path), _ROOT)
    session.log(f"built wheel in {dist_path}")


@nox.session
def test_wheel(session: nox.Session):
    session.notify("build_wheel")
    session.notify("test_wheel_with_python-3.8")
    session.notify("test_wheel_with_python-3.9")
    session.notify("test_wheel_with_python-3.10")
    session.notify("test_wheel_with_python-3.11")


@nox.session(python=["3.8", "3.9", "3.10", "3.11"])
def test_wheel_with_python(session: nox.Session):
    wheel_path = next(session.cache_dir.joinpath("dist").glob("gridtools_cpp-*.whl"))
    session.install("pytest")
    session.install(str(wheel_path))
    session.run("pytest", "tests", *session.posargs)


@nox.session
def clean(session: nox.Session):
    shutil.rmtree(session.cache_dir.joinpath("dist"), True)
