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

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import random
from jumanji.testing.env_not_smoke import check_env_does_not_smoke
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep

from matrax import MatrixGame, State
from matrax.games import climbing_game


@pytest.fixture
def matrix_game_env() -> MatrixGame:
    """Instantiates a default MatrixGame environment with no state."""
    # use climbing game for tests
    env = MatrixGame(climbing_game, keep_state=False)
    return env


@pytest.fixture
def matrix_game_env_with_state() -> MatrixGame:
    """Instantiates a default MatrixGame environment that keeps state."""
    # use climbing game for tests
    env = MatrixGame(climbing_game, keep_state=True)
    return env


def test_matrix_game__specs(matrix_game_env: MatrixGame) -> None:
    """Validate environment specs conform to the expected shapes and values"""
    action_spec = matrix_game_env.action_spec()
    observation_spec = matrix_game_env.observation_spec()

    assert observation_spec.agent_obs.shape == (2, 2)  # type: ignore
    assert action_spec.num_values.shape[0] == matrix_game_env.num_agents
    assert action_spec.num_values[0] == matrix_game_env.num_actions


def test_matrix_game__reset(matrix_game_env: MatrixGame) -> None:
    """Validate the jitted reset of the environment."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(matrix_game_env.reset, n=1))

    key1, key2 = random.PRNGKey(0), random.PRNGKey(1)
    state1, timestep1 = reset_fn(key1)
    state2, timestep2 = reset_fn(key2)

    assert isinstance(timestep1, TimeStep)
    assert isinstance(state1, State)
    assert state1.step_count == 0

    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # reset function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(state1)
    # Check random initialization
    assert not jnp.all(state1.key == state2.key)
    assert state1.step_count == state2.step_count


def test_matrix_game__agent_observation(
    matrix_game_env: MatrixGame, matrix_game_env_with_state: MatrixGame
) -> None:
    """Validate the agent observation function."""
    # validate starting observation
    state, timestep = matrix_game_env.reset(random.PRNGKey(0))
    assert jnp.array_equal(timestep.observation.agent_obs, jnp.array([[0, 0], [0, 0]]))

    # validate zero observation when not using previous actions as observations
    state, timestep = matrix_game_env.step(state, jnp.array([1, 0]))
    assert jnp.array_equal(timestep.observation.agent_obs, jnp.array([[0, 0], [0, 0]]))

    # validate starting observation when using previous actions as observations
    state, timestep = matrix_game_env_with_state.reset(random.PRNGKey(0))
    assert jnp.array_equal(
        timestep.observation.agent_obs, jnp.array([[-1, -1], [-1, -1]])
    )

    # validate actions observation when using previous actions as observations
    state, timestep = matrix_game_env_with_state.step(state, jnp.array([1, 0]))
    assert jnp.array_equal(timestep.observation.agent_obs, jnp.array([[1, 0], [1, 0]]))


def test_matrix_game__step(matrix_game_env_with_state: MatrixGame) -> None:
    """Validate the jitted step function of the environment."""
    chex.clear_trace_counter()

    step_fn = chex.assert_max_traces(matrix_game_env_with_state.step, n=1)
    step_fn = jax.jit(step_fn)

    state, timestep = matrix_game_env_with_state.reset(random.PRNGKey(10))

    action1 = jnp.array([0, 0])
    action2 = jnp.array([0, 1])

    new_state1, timestep1 = step_fn(state, action1)

    # Check that rewards have the correct number of dimensions
    assert jnp.ndim(timestep1.reward) == 1
    assert jnp.ndim(timestep.reward) == 0
    # Check that discounts have the correct number of dimensions
    assert jnp.ndim(timestep1.discount) == 0
    assert jnp.ndim(timestep.discount) == 0
    # Check that the state is made of DeviceArrays, this is false for the non-jitted
    # step function since unpacking random.split returns numpy arrays and not device arrays.
    assert_is_jax_array_tree(new_state1)
    # Check that the state has changed
    assert new_state1.step_count != state.step_count
    assert not jnp.array_equal(
        timestep1.observation.agent_obs, timestep.observation.agent_obs
    )
    # Check that two different actions lead to two different states
    _, timestep2 = step_fn(state, action2)
    assert not jnp.array_equal(
        timestep1.observation.agent_obs, timestep2.observation.agent_obs
    )


def test_matrix_game__does_not_smoke(matrix_game_env: MatrixGame) -> None:
    """Validate that we can run an episode without any errors."""
    check_env_does_not_smoke(matrix_game_env)


def test_matrix_game__time_limit(matrix_game_env: MatrixGame) -> None:
    """Validate the termination after time limit has been reached."""
    step_fn = jax.jit(matrix_game_env.step)
    state_key = random.PRNGKey(10)
    state, timestep = matrix_game_env.reset(state_key)
    assert timestep.first()

    for _ in range(matrix_game_env.time_limit - 1):
        state, timestep = step_fn(state, jnp.array([0, 0]))

    assert timestep.mid()
    state, timestep = step_fn(state, jnp.array([0, 0]))
    assert timestep.last()


def test_matrix_game__reward(matrix_game_env: MatrixGame) -> None:
    """Validate the rewards are correct based on agent actions."""
    step_fn = jax.jit(matrix_game_env.step)
    state_key = random.PRNGKey(10)
    state, timestep = matrix_game_env.reset(state_key)

    state, timestep = step_fn(state, jnp.array([0, 0]))
    jax.debug.print("rewards: {r}", r=timestep.reward)
    assert jnp.array_equal(timestep.reward, jnp.array([11, 11]))

    state, timestep = step_fn(state, jnp.array([1, 0]))
    assert jnp.array_equal(timestep.reward, jnp.array([-30, -30]))

    state, timestep = step_fn(state, jnp.array([0, 1]))
    assert jnp.array_equal(timestep.reward, jnp.array([-30, -30]))

    state, timestep = step_fn(state, jnp.array([1, 1]))
    assert jnp.array_equal(timestep.reward, jnp.array([7, 7]))

    state, timestep = step_fn(state, jnp.array([1, 2]))
    assert jnp.array_equal(timestep.reward, jnp.array([0, 0]))

    state, timestep = step_fn(state, jnp.array([2, 2]))
    assert jnp.array_equal(timestep.reward, jnp.array([5, 5]))
