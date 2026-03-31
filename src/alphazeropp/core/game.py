import copy
from abc import ABC, abstractmethod
from typing import Tuple, Any, TypeVar, Generic, Hashable

from gymnasium import Env, Space
import numpy as np
from numpy.typing import ArrayLike

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class Game(ABC, Generic[ObsType, ActType]):
    """
    Abstract base class for single-player games compatible with this package's AlphaZero
    implementation. Written so that a Farama Gymnasium will automatically be a full
    implementation. In addition to implementing the abstract methods, the class must define
    fields `action_space` and `observation_space`.
    """

    obs: ObsType | None
    reward: float | None
    terminated: bool | None
    truncated: bool | None
    info: dict[str, Any] | None
    step_count: int | None

    def __init__(self):
        self.obs = None
        self.reward = None
        self.terminated = None
        self.truncated = None
        self.info = None
        self.step_count = None

    @abstractmethod
    def step(self, action: ActType) -> Tuple[ObsType, Any, bool, bool, dict[str, Any]]:
        pass

    @abstractmethod
    def reset(self) -> Tuple[ObsType, dict[str, Any]]:
        pass

    @abstractmethod
    def get_action_mask(self) -> Any:
        pass

    @classmethod
    def _hashable_obs_impl(cls, obs) -> Hashable:
        # PERF this might be rather non-performant, seeking generality right now
        try:
            hash(obs)
            return obs
        except TypeError:
            if isinstance(obs, list) or isinstance(obs, tuple):
                return tuple(cls._hashable_obs_impl(sub_obs) for sub_obs in obs)
            try:
                arr = np.asarray(obs)
                return arr.tobytes()
            except:
                raise TypeError(f"Generic implementation of `_hashable_obs_impl` failed to make observation {obs} hashable. Please implement your own `_hashable_obs_impl` classmethod or `hashable_obs` property")
    
    @property
    def hashable_obs(self) -> Hashable:
        "Returns a hashable representation of the current observation `obs`."
        return self._hashable_obs_impl(self.obs)

    def reset_wrapper(self, **kwargs):
        obs, info = self.reset(**kwargs)
        self.obs = obs
        self.reward = None
        self.terminated = None
        self.truncated = None
        self.info = info
        self.step_count = 0
        return obs, info
    
    def step_wrapper(self, action: ActType) -> Tuple[ObsType, Any, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.step(action)
        self.obs = obs
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
        self.step_count = self.step_count + 1 if self.step_count is not None else 0
        return obs, reward, terminated, truncated, info
    
    def stash_state(self) -> Any:
        """
        Returns a representation of the game state such that the game can be restored later
        after step() has been called several times. The default implementation relies on
        `deepcopy`, subclasses may wish to provide a more performant implementation. If
        overriding, you must handle self.obs, self.reward, self.terminated, self.truncated,
        self.info, and self.step_count.
        """
        return copy.deepcopy(self)

    def unstash_state(self, state: Any):
        """
        Reverts this game to the state represented by `state` (from `stash_state`).
        Modifies self in-place and returns self.  Override this if you override
        `stash_state`.
        """
        self.__dict__.update(state.__dict__)
        return self

    def clone(self) -> "Game":
        """Return an independent playable copy of this game."""
        return copy.deepcopy(self)

class EnvGame(Game[ObsType, ActType]):
    """
    Abstract base class for Farama Gymnasium-based games. Contains a field `env:
    Env[ObsType, ActType]` populated in the constructor; implements `step` and `reset`
    methods that wrap `env`'s `step` and `reset`.
    """

    env: Env[ObsType, ActType]
    action_space: Space[ActType]
    observation_space: Space[ObsType]

    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action: ActType) -> Tuple[ObsType, Any, bool, bool, dict[str, Any]]:
        return self.env.step(action)
    
    def reset(self, **kwargs) -> Tuple[ObsType, dict[str, Any]]:
        return self.env.reset( **kwargs)
    
    def get_action_mask(self, *args, **kwargs):
        try:
            return self.env.get_action_mask(*args, **kwargs)  # type: ignore
        except AttributeError:
            raise ValueError("Environment must implement `get_action_mask`")
    
    def render(self, *args, **kwargs):
        "For convenience, we provide a `render` method that calls `env`'s `render`."
        return self.env.render(*args, **kwargs)