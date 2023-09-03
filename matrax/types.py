# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex


@dataclass
class State:
    """A dataclass representing the state of the game.
    step_count: an integer representing the current step of the episode.
    key: a pseudorandom number generator key.
    """

    step_count: chex.Array  # ()
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    """The observation that the agent sees.
    agent_obs: the agent's observation.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agent_obs: chex.Array  # (num_agents,)
    step_count: chex.Array  # ()
