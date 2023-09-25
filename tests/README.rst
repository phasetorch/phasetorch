Tests
=====

Some helpful ``pytest`` terminal commands are given below -

* ``pytest`` runs all tests under the current directory recursively. A test is any function or method that begins with ``test``.
* ``pytest -s`` runs tests while also displaying the outputs from the ``print`` statement.
* ``pytest -k "nlpr"`` runs only those tests whose function name include ``nlpr``.
* ``pytest -rsx`` shows why a test was skipped.
* ``python pyshow.py --file <path-to-image-file> --vmin <min-scale> --vmax <max-scale>`` is useful to display images files.
* ``requirements.txt`` in the top-level directory contains the packages inside a conda environment called ``phasetorch_pytest`` for CI testing. 
