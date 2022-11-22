# GridTools
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import nox

nox.options.sessions = ["tests", "test_wheel"]


def get_wheel(session: nox.Session) -> pathlib.Path:
    return list(session.cache_dir.joinpath("dist").glob("gridtools-*.whl"))[0]


@nox.session
def build_wheel(session: nox.Session):
    dist_path = session.cache_dir.joinpath("dist").absolute()
    workdir = pathlib.Path(".").absolute()
    session.install("pip")
    with session.chdir(session.cache_dir):
        session.run("pip", "wheel", "-w", str(dist_path), str(workdir))
        session.log(f"built wheel in {dist_path}")
    session.notify("test_wheel")


@nox.session
def tests(session: nox.Session):
    session.install(".")
    session.install("pytest")
    session.run("pytest", "tests", *session.posargs)


@nox.session
def test_wheel(session: nox.Session):
    build_wheel(session)
    wheel_path = get_wheel(session)
    session.install("pytest")
    session.install(str(wheel_path))
    session.run("pytest", "tests", *session.posargs)
