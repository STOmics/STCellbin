# Copyright (C) 2023 - 2025 BGI Research.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import enum


class IPR(object):
    def __init__(self):
        self.ipr_version: str = ''


class IPRVersion(enum.Enum):
    V0D1D0 = '0.1.0'
    V0D0D1 = '0.0.1'
    V0D1D2 = '0.1.2'
    UNKNOWN = 'unknown'
