# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A LazyLoader class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import types
import logging


# Note: Prior to adopting Tensorflow's LazyLoader class, we tried instead to use the `lazy_import` magic function,
# as detailed in https://docs.python.org/3/library/importlib.html#implementing-lazy-imports.
#
# However, this function was a little too "magical": There were quite a few behind-the-scenes side effects that made
# the lazy behavior unpredictable:
#   - Importing a parent module would throw an `AttributeNotFoundError` when trying to access submodules/classes/etc.
#   - Importing a submodule instead (e.g. `lazy_import("sklearn.metrics")` would inadvertently trigger imports in the
#     parent module during the call to `loader.exec_module(module)`.
#   - There seemed to be complex side effects for our deep learning packages, especially those that depended on `torch`.
#     Lazy loading `torch` seemed to cause obscure errors for other packages such as `monai`.
#
# Because we are trying to lazy load deep learning packages, it made sense to look to the Python DL ecosystem for
# lazy loading implementations. And, luckily, Tensorflow has an incredibly straightforward implementation that removes
# much of the magic that comes with the `ModuleSpec` and `LazyLoader` classes from importlib.
#
# This class is also in use as-is in many downstream DL packages, signifying its reliability. For example:
#  - https://github.com/bentoml/BentoML/blob/df2c7b86decbdff053002bddf7beb228d012a895/src/bentoml/_internal/utils/lazy_loader.py

# FIXME: The LazyLoader class only allows you to access attributes of the lazy-loaded module (functions, classes,
#  constants). It *DOES NOT* allow you to access submodules. So, make sure to lazy-load the lowest-level module needed.

class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies.

    `contrib`, and `ffmpeg` are examples of modules that are large and not always
    needed, and this allows them to only be loaded when they are used.
    """

    # The lint error here is incorrect.
    def __init__(self, local_name, parent_module_globals, name, warning=None):  # pylint: disable=super-on-old-class
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._warning = warning

        super(LazyLoader, self).__init__(name)

    def _load(self):
        """Load the module and insert it into the parent's globals."""
        # Import the target module and insert it into the parent's namespace
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

        # Emit a warning if one was specified
        if self._warning:
            logging.warning(self._warning)
            # Make sure to only warn once.
            self._warning = None

        # Update this object's dict so that if someone keeps a reference to the
        #     LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #     that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)
