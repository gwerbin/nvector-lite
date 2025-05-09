from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("reload_ext",  "autoreload")
ipython.run_line_magic("aimport", "nvector_lite")
ipython.run_line_magic("autoreload", "2")
