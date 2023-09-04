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

import chex
import jax.numpy as jnp


def convert_payoff_vector_to_matrix(payoff_vec: list) -> chex.Array:
    payoff_matrix = jnp.array(
        [
            [[payoff_vec[0], payoff_vec[2]], [payoff_vec[4], payoff_vec[6]]],
            [[payoff_vec[1], payoff_vec[3]], [payoff_vec[5], payoff_vec[7]]],
        ]
    )
    return payoff_matrix
