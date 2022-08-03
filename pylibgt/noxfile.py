import nox

nox.options.sessions = ["tests"]


@nox.session
def tests(session):
    session.install(".")
    session.install("pytest")
    session.run("pytest", "tests", *session.posargs)
