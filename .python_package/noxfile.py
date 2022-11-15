# GridTools
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import pathlib
import shutil

import nox

nox.options.sessions = ["tests", "test_wheel"]


def get_wheel(session: nox.Session) -> pathlib.Path:
    return list(session.cache_dir.joinpath("dist").glob("gridtools-*.whl"))[0]


@nox.session(tags=["build"])
def build_wheel(session: nox.Session):
    dist_path = session.cache_dir.joinpath("dist").absolute()
    workdir = pathlib.Path(".").absolute()
    session.install("pip")
    with session.chdir(session.cache_dir):
        session.run("pip", "wheel", "-w", str(dist_path), str(workdir))
    session.notify("test_wheel")


@nox.session
def reuse_wheel(session: nox.Session):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-wheel", action="store_true")
    args = parser.parse_args(args=session.posargs)

    session.log("checking for cached wheel")
    try:
        get_wheel(session)
        session.log("cached wheel found")
    except IndexError:
        session.log("cached wheel not found")
        build_wheel(session)
    else:
        if args.rebuild_wheel:
            session.log("wheel rebuild requested")
            session.posargs.remove("--rebuild-wheel")
            build_wheel(session)


@nox.session(tags=["test"])
def tests(session: nox.Session):
    session.install(".")
    session.install("pytest")
    session.run("pytest", "tests", *session.posargs)


@nox.session(tags=["test"])
def test_wheel(session: nox.Session):
    reuse_wheel(session)
    wheel_path = get_wheel(session)
    session.install("pytest")
    session.install(str(wheel_path))
    session.run("pytest", "tests", *session.posargs)


@nox.session(tags=["build"])
def copy_wheel(session: nox.Session):
    reuse_wheel(session)
    wheel_path = get_wheel(session)
    dist_path = pathlib.Path("./dist")
    dist_path.mkdir(exist_ok=True)
    shutil.copy(wheel_path, dist_path)
