.. include:: ../defs.hrst

.. _faq:

=======================================
Frequently Asked Questions
=======================================

How do I build the docs?
------------------------
The documentation is based on sphinx. You need at least::

    pip install sphinx
    pip install cloud-sptheme
    pip install sphinxcontrib-fulltoc

Go to ``docs/manuals`` and::

    make html

The output will be in ``_build/html``

