"""
banana

Copyright (c) 2012-2018 Thomas G. Close, Monash Biomedical Imaging,
Monash University, Melbourne, Australia

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os.path as op

RESOURCES_DIR = op.join(op.realpath(op.abspath(op.dirname(__file__))),
                        'resources')

MATLAB_RESOURCES = op.join(RESOURCES_DIR, 'matlab')
