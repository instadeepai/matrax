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

import functools
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition

from matrax.types import Observation, State


class MatrixGame(Environment[State]):
    """JAX implementation of the 2-player matrix game environment:
    https://github.com/uoe-agents/matrix-games

    A matrix game is a two-player game where each player has a set of actions and a payoff matrix.
    The payoff matrix is a 2D array where the rows correspond to the actions of player 1 and the
    columns correspond to the actions of player 2. The entry at row i and column j for player 1 is
    the reward given to player 1 when playing action i and player 2 plays action j. Similarly, the
    entry at row i and column j for player 2 is the reward given to player 2 when playing action j
    and player 1 plays action i.

    ```python
    from matrax import MatrixGame
    payoff_matrix = jnp.array([[1, -1], [-1, 1]])
    env = MatrixGame(payoff_matrix)
    key = jax.random.PRNGKey(0)
    state, timestep = jax.jit(env.reset)(key)
    action = env.action_spec().generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    ```
    """

    def __init__(
        self,
        payoff_matrix: chex.Array,
        keep_state: bool = True,
        time_limit: int = 500,
    ):
        """Instantiates a `MatrixGame` environment.

        Args:
            payoff_matrix: a 2D array of shape (num_agents, num_agents) containing the payoff
                matrix of the game.
            keep_state: whether to keep state by giving agents the actions of all players in
                the previous round as observations. Defaults to True.
            time_limit: the maximum step limit allowed within the environment.
                Defaults to 500.
        """
        self.payoff_matrix = payoff_matrix
        self.keep_state = keep_state

        self.num_agents = self.payoff_matrix.shape[0]
        self.num_actions = self.payoff_matrix.shape[1]

        self.time_limit = time_limit

    def __repr__(self) -> str:
        return f"MatrixGame(\n" f"\tpayoff_matrix={self.payoff_matrix!r},\n" ")"

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Resets the environment.

        Args:
            key: random key used to reset the environment since it is stochastic.

        Returns:
            state: State object corresponding to the new state of the environment.
            timestep: TimeStep object corresponding the first timestep returned by the environment.
        """
        # create environment state
        state = State(
            step_count=jnp.array(0, int),
            key=key,
        )
        dummy_actions = jnp.ones((self.num_agents,), int) * -1

        # collect first observations and create timestep
        agent_obs = jax.vmap(
            functools.partial(self._make_agent_observation, dummy_actions)
        )(jnp.arange(self.num_agents))
        observation = Observation(
            agent_obs=agent_obs,
            step_count=state.step_count,
        )
        timestep = restart(observation=observation)
        return state, timestep

    def step(
        self,
        state: State,
        actions: chex.Array,
    ) -> Tuple[State, TimeStep[Observation]]:
        """Perform an environment step.

        Args:
            state: State object containing the dynamics of the environment.
            actions: Array containing actions of each agent.

        Returns:
            state: State object corresponding to the next state of the environment.
            timestep: TimeStep object corresponding the timestep returned by the environment.
        """

        def compute_reward(
            actions: chex.Array, payoff_matrix_per_agent: chex.Array
        ) -> chex.Array:
            reward_idx = tuple(actions)
            return payoff_matrix_per_agent[reward_idx]

        rewards = jax.vmap(functools.partial(compute_reward, actions))(
            self.payoff_matrix
        )

        # construct timestep and check environment termination
        steps = state.step_count + 1
        done = steps >= self.time_limit

        # compute next observation
        agent_obs = jax.vmap(functools.partial(self._make_agent_observation, actions))(
            jnp.arange(self.num_agents)
        )
        next_observation = Observation(
            agent_obs=agent_obs,
            step_count=steps,
        )

        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            rewards,
            next_observation,
        )

        # create environment state
        next_state = State(
            step_count=steps,
            key=state.key,
        )
        return next_state, timestep

    def _make_agent_observation(
        self,
        actions: chex.Array,
        agent_id: int,
    ) -> chex.Array:
        return jax.lax.cond(
            self.keep_state,
            lambda: actions,
            lambda: jnp.zeros(self.num_agents, int),
        )

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the MatrixGame environment.
        Returns:
            Spec for the `Observation`, consisting of the fields:
                - agent_obs: BoundedArray (int32) of shape (num_agents, num_agents).
                - step_count: BoundedArray (int32) of shape ().
        """

        obs_shape = (self.num_agents, self.num_agents)
        low = jnp.zeros(obs_shape)
        high = jnp.ones(obs_shape) * self.num_actions

        agent_obs = specs.BoundedArray(obs_shape, jnp.int32, low, high, "agent_obs")
        step_count = specs.BoundedArray((), jnp.int32, 0, self.time_limit, "step_count")
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_obs=agent_obs,
            step_count=step_count,
        )

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec.
        Since this is a multi-agent environment, the environment expects an array of actions.
        This array is of shape (num_agents,).
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([self.num_actions] * self.num_agents, jnp.int32),
            name="action",
        )
