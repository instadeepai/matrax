# Copyright 2023 InstaDeep Ltd. All rights reserved.
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

from jumanji.registration import make, register
from jumanji.version import __version__

from matrax.env import MatrixGame
from matrax.games import climbing_game, conflict_games, no_conflict_games, penalty_games
from matrax.types import Observation, State

"""Environment Registration"""

# Penalty games
for penalty_value, payoff_matrix in penalty_games.items():
    register(
        id=f"Penalty-{penalty_value}-stateless-v0",
        entry_point="matrax:MatrixGame",
        kwargs={
            "payoff_matrix": payoff_matrix,
            "keep_state": False,
        },
    )
    register(
        id=f"Penalty-{penalty_value}-stateful-v0",
        entry_point="matrax:MatrixGame",
        kwargs={
            "payoff_matrix": payoff_matrix,
            "keep_state": True,
            "time_limit": 1000,
        },
    )

# climbing game
register(
    id="Climbing-stateless-v0",
    entry_point="matrax:MatrixGame",
    kwargs={
        "payoff_matrix": climbing_game,
        "keep_state": False,
    },
)
register(
    id="Climbing-stateful-v0",
    entry_point="matrax:MatrixGame",
    kwargs={
        "payoff_matrix": climbing_game,
        "keep_state": True,
        "time_limit": 1000,
    },
)

# no conflict games
for _id, payoff_matrix in no_conflict_games.items():
    register(
        id=f"NoConflict-{_id}-stateless-v0",
        entry_point="matrax:MatrixGame",
        kwargs={
            "payoff_matrix": payoff_matrix,
            "keep_state": False,
        },
    )
    register(
        id=f"NoConflict-{_id}-stateful-v0",
        entry_point="matrax:MatrixGame",
        kwargs={
            "payoff_matrix": payoff_matrix,
            "keep_state": True,
            "time_limit": 1000,
        },
    )

# conflict games
for _id, payoff_matrix in conflict_games.items():
    register(
        id=f"Conflict-{_id}-stateless-v0",
        entry_point="matrax:MatrixGame",
        kwargs={
            "payoff_matrix": payoff_matrix,
            "keep_state": False,
        },
    )
    register(
        f"Conflict-{_id}-stateful-v0",
        entry_point="matrix:MatrixGame",
        kwargs={
            "payoff_matrix": payoff_matrix,
            "keep_state": True,
            "time_limit": 1000,
        },
    )
