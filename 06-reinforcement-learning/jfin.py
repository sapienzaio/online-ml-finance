import jax
import chex
import jax.numpy as jnp
from typing import Callable
from functools import partial


@chex.dataclass
class BanditState:
    count: jax.Array
    value: jax.Array
    key: jax.random.PRNGKey
    
    def update(self, action, value, count):
        new_key = jax.random.split(self.key)[1]
        state = self.replace(
            value=self.value.at[action].set(value),
            count=self.count.at[action].set(count),
            key=new_key,
        )
        return state


class EpsilonGreedyAgent:
    def __init__(self, epsilon, alpha = 0.0):
        self.epsilon = epsilon
        self.alpha = alpha
    
    def _step(self, state, rewards, storefn):
        key_choice, key_arm = jax.random.split(state.key)
        is_greedy = jax.random.bernoulli(key_choice, p=1 - self.epsilon) # w.r.t. the value function
        n_bandits = state.value.shape[0]
        random_choice = jax.random.choice(key_arm, n_bandits)
        
        action = state.value.argmax() * is_greedy + random_choice * (1 - is_greedy) 
        reward = rewards[action]
        new_count = state.count[action] + 1
        
        discount = (1 / new_count) * (self.alpha == 0.0) + self.alpha
        new_value = state.value[action] + (reward - state.value[action]) * discount
        
        state = state.update(action, new_value, new_count)
        return state, storefn(state, action, reward)
    
    def init(self, key, n_arms):
        count_v = jnp.zeros(n_arms)
        value_v = jnp.zeros(n_arms)
        state = BanditState(count=count_v, value=value_v, key=key)
        return state


    @partial(jax.jit, static_argnames=("self", "storefn",))
    def init_and_run(self, key, data, storefn):
        state = self.init(key, data.shape[1])
        partial_step = partial(self._step,  storefn=storefn)
        state, hist = jax.lax.scan(partial_step, state, data)
        return state, hist
    

    @partial(jax.jit, static_argnames=("self", "storefn", "n_sims"))
    def init_and_run_sims(self, key, data, storefn, n_sims):
        keys = jax.random.split(key, n_sims)
        vmap_returns = jax.vmap(self.init_and_run, in_axes=(0, None, None))
        state, hist = vmap_returns(keys, data, storefn)
        return state, hist
    



@partial(jax.jit, static_argnames=("n_sims", "storefn"))
def run_bandit_sims(key, data, epsilon, alpha, n_sims, storefn):
    keys = jax.random.split(key, n_sims)
    
    vmap_returns = jax.vmap(run_bandit, in_axes=(0, None, None, None, None))
    _, hist = vmap_returns(keys, data, epsilon, alpha, storefn)
    return hist


@partial(jax.jit, static_argnames=("n_sims",))
@partial(jax.vmap, in_axes=(None, None, 0, None, None))
def run_bandit_sims_eps(key, data, epsilon_v, alpha, n_sims):
    return run_bandit_sims(key, data, epsilon_v, alpha, n_sims, store_reward)