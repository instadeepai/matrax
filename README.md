<p align="center">
    <a href="docs/img/matrax_logo.png">
        <img src="docs/img/matrax_logo.png" alt="Matrax logo" width="50%"/>
    </a>
</p>

<h2 align="center">
    <p>Matrix Games in JAX</p>
</h2>
<p align="center">
    <a href="https://www.python.org/doc/versions/">
      <img src="https://img.shields.io/pypi/pyversions/jumanji.svg?style=flat-square" alt="Python Versions">
    </a>
    <a href="https://badge.fury.io/py/matrax">
      <img src="https://badge.fury.io/py/matrax.svg" alt="PyPI version" height="18">
    </a>
    <a href="https://github.com/instadeepai/jumanji/actions/workflows/tests_linters.yml">
      <img src="https://github.com/instadeepai/jumanji/actions/workflows/tests_linters.yml/badge.svg" alt="Tests">
    </a>
    <a href="https://github.com/psf/black">
      <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style">
    </a>
    <a href="http://mypy-lang.org/">
      <img src="http://www.mypy-lang.org/static/mypy_badge.svg" alt="MyPy">
    </a>
    <a href="https://opensource.org/licenses/Apache-2.0">
      <img src="https://img.shields.io/badge/License-Apache%202.0-orange.svg" alt="License">
    </a>
</p>

## Enter the Matrax! 😎

<div align="center">
<h3>

[**Installation**](#installation-) | [**Quickstart**](#quickstart-)

</div>

**Matrax** is a lightweight suite of [2-player matrix game](https://en.wikipedia.org/wiki/Normal-form_game) environments written in [JAX](https://github.com/google/jax). It is a direct re-implementation of the matrix games provided in [this repository](https://github.com/uoe-agents/matrix-games) from the [AARG](https://agents.inf.ed.ac.uk/). It follows the [Jumanji](https://github.com/instadeepai/jumanji) RL environment suite API developed by [InstaDeep](https://www.instadeep.com/).

<h2 name="environments" id="environments">2-Player Matrix Games 🧑‍🤝‍🧑 </h2>

| Category                              | Shape (action space) | Registered Version(s)                                | Source                                                                                           |
|------------------------------------------|----------|------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| 🔻 Penalty Game                              | 3 x 3  | `Penalty-{k}-{state}-v0`                                        | [code](https://github.com/instadeepai/matrax/blob/main/matrax/games/penalty.py)   |
| 🧗‍♀️ Climbing Game                              | 3 x 3  | `Climbing-{state}-v0`                                        | [code](https://github.com/instadeepai/matrax/blob/main/matrax/games/climbing.py)   |
| 🤝 No-Conflict Games                              | 2 x 2  | `NoConflict-{id}-{state}-v0`                                   | [code](https://github.com/instadeepai/matrax/blob/main/matrax/games/no_conflict.py)   |
| 💣 Conflict Games                        | 2 x 2    | `Conflict-{id}-{state}-v0`                                     | [code](https://github.com/instadeepai/matrax/blob/main/matrax/games/conflict.py) |

<h2 name="install" id="install">Installation 🎬</h2>

You can install the latest release of **Matrax** from PyPI:

```bash
pip install matrax
```

Alternatively, you can install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/instadeepai/matrax.git
```

**Matrax** has been tested on Python 3.8 and 3.9.
Note that because the installation of JAX differs depending on your hardware accelerator,
we advise users to explicitly install the correct JAX version (see the
[official installation guide](https://github.com/google/jax#installation)).

<h2 name="quickstart" id="quickstart">Quickstart ⚡</h2>

```python
import jax
import matrax

# Instantiate a matrix game environment using the registry
env = matrax.make("Penalty-25-stateless-v0")

# Reset your (jit-able) environment
key = jax.random.PRNGKey(0)
state, timestep = jax.jit(env.reset)(key)

# Interact with the (jit-able) environment
action = env.action_spec().generate_value()          # Action selection (dummy value here)
state, timestep = jax.jit(env.step)(state, action)   # Take a step and observe the next state and time step
```

### Registry and Versioning 📖

Like Jumanji, **Matrax** keeps a strict versioning of its environments for reproducibility reasons.
We maintain a registry of standard environments with their configuration.
For each environment, a version suffix is appended, e.g. `EnvironmentName-v1`.
When changes are made to environments that might impact learning results,
the version number is incremented by one to prevent potential confusion.

## See Also 🔎

Other works have embraced the approach of writing RL environments in JAX.
In particular, we suggest users check out the following sister repositories:

- 🌴 [Jumanji](https://github.com/instadeepai/jumanji) is a diverse suite of scalable reinforcement learning environments.
- 🦾 [Brax](https://github.com/google/brax) is a differentiable physics engine that simulates
environments made up of rigid bodies, joints, and actuators.
- 🏋️‍ [Gymnax](https://github.com/RobertTLange/gymnax) implements classic environments including
classic control, bsuite, MinAtar and a collection of meta RL tasks.
- 🎲 [Pgx](https://github.com/sotetsuk/pgx) provides classic board game environments like
Backgammon, Shogi, and Go.
