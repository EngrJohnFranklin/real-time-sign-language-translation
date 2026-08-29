"""
Pytest Configuration
Shared fixtures, mocks, and test configurations
"""

import os
import sys

# Ensure TCL/TK libraries can be located when running tests
tcl_dir = os.path.join(sys.base_prefix, "tcl", "tcl8.6")
tk_dir = os.path.join(sys.base_prefix, "tcl", "tk8.6")
if os.path.exists(tcl_dir) and "TCL_LIBRARY" not in os.environ:
    os.environ["TCL_LIBRARY"] = tcl_dir
if os.path.exists(tk_dir) and "TK_LIBRARY" not in os.environ:
    os.environ["TK_LIBRARY"] = tk_dir

