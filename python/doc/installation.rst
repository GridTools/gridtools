Installation
============

.. highlight:: bash

Quick installation with Vagrant
-------------------------------

Vagrant can be used to quickly setup a portable development environment.

*   Download and install Vagrant for your OS (available
    `here <https://www.vagrantup.com/downloads.html>`_).
*   Download and install VirtualBox version 5 or later (available
    `here <https://www.virtualbox.org/wiki/Downloads>`_).
*   Install the Vagrant ``vbguest`` plugin by calling ```vagrant plugin install
    vagrant-vbguest``. This is necessary to enable folder synchronization using
*   VirtualBox shared folders.
*   Go to the GridTools top directory and make sure the
    files ``Vagrant_bootstrap.sh`` and ``Vagrantfile`` exist.
*   From the command line call ``vagrant up`` in order to bootstrap the Debian-based
    Vagrant VM. This step will take some time to complete, so make sure you don't
    do this while running on battery.
*   To connect to the newly-created Vagrant VM, use ``vagrant ssh``.
*   Once logged into Vagrant, issue these commands to launch the demo IPython
    notebook, e.g.::

        $> source venv/bin/activate
        (venv) $> cd /vagrant/python/samples/
        (venv) $> ipython notebook PASC16.ipynb

*   To shutdown Vagrant, logout of the `ssh` session and issue ``vagrant halt``.
*   You may check the status of the Vagrant VM at any time by executing ``vagrant status``.


.. warning::
   The Vagrant directory ``/vagrant`` is rsync-ed on every ``vagrant up`` so you
   may potentially lose data in case you simultaneously modify its contents from
   inside the VM and the host.


Native installation
-------------------
