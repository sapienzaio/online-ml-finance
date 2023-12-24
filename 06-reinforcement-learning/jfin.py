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
    
    def update(self, ix, value, count):
        new_key = jax.random.split(self.key)[1]
        state = self.replace(
            value=self.value.at[ix].set(value),
            count=self.count.at[ix].set(count),
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
        n_bandits = state.value.shape[-1]
        random_choice = jax.random.choice(key_arm, n_bandits)
        
        action = state.value.argmax() * is_greedy + random_choice * (1 - is_greedy) 
        reward = rewards[action]
        new_count = state.count[action] + 1
        
        discount = (1 / new_count) * (self.alpha == 0.0) + self.alpha
        new_value = state.value[action] + (reward - state.value[action]) * discount
        
        state = state.update(action, new_value, new_count)
        carry = storefn(state, action, reward)

        return state, carry
    
    def init(self, key, n_arms):
        count_v = jnp.zeros(n_arms)
        value_v = jnp.zeros(n_arms)
        state = BanditState(count=count_v, value=value_v, key=key)
        return state

    @partial(jax.jit, static_argnames=("self", "storefn",))
    def init_and_run(self, key, rewards, storefn):
        state = self.init(key, rewards.shape[1])
        partial_step = partial(self._step,  storefn=storefn)
        state, hist = jax.lax.scan(partial_step, state, rewards)
        return state, hist
    

    @partial(jax.jit, static_argnames=("self", "storefn", "n_sims"))
    def init_and_run_sims(self, key, rewards, storefn, n_sims):
        keys = jax.random.split(key, n_sims)
        vmap_returns = jax.vmap(self.init_and_run, in_axes=(0, None, None))
        state, hist = vmap_returns(keys, rewards, storefn)
        return state, hist
    

class TabularEpsilonGreedyAgent:
    def __init__(self, epsilon, alpha = 0.0):
        self.epsilon = epsilon
        self.alpha = alpha

    def init(self, key, n_arms, n_options):
        """
        Parameters
        key: jax.random.PRNGKey
        n_arms: int
            Number of arms
        n_options: int
            Number of states (contexts) per arm
        """
        shape = (*(n_options,) * n_arms, n_arms)
        value_v = jnp.zeros(shape)
        state = BanditState(value=value_v, count=value_v, key=key)
        return state

    def _step(self, state, xs, storefn):
        rewards, context = xs
        context = tuple(context)
        key_choice, key_arm = jax.random.split(state.key)

        is_greedy = jax.random.bernoulli(key_choice, p=1-self.epsilon)
        n_bandits = state.value.shape[-1]
        random_choice = jax.random.choice(key_arm, n_bandits)
        
        action = state.value[context].argmax() * is_greedy + random_choice * (1 - is_greedy) 
        reward = rewards[action]

        update_ix = tuple((*context, action))
        new_count = state.count[update_ix] + 1
        discount = (1 / new_count) * (self.alpha == 0.0) + self.alpha
        new_value = state.value[update_ix] + discount * (reward - state.value[update_ix])

        state = state.update(update_ix, new_value, new_count)
        carry = storefn(state, action, reward)
        
        return state, carry 


    @partial(jax.jit, static_argnames=("self", "storefn",))
    def init_and_run(self, key, rewards, contexts, storefn):
        xs = (rewards, contexts)
        state = self.init(key, rewards.shape[1], n_options=2)

        partial_step = partial(self._step,  storefn=storefn)
        state, hist = jax.lax.scan(partial_step, state, xs)
        return state, hist


    @partial(jax.jit, static_argnames=("self", "storefn", "n_sims"))
    def init_and_run_sims(self, key, rewards, context, storefn, n_sims):
        keys = jax.random.split(key, n_sims)
        vmap_returns = jax.vmap(self.init_and_run, in_axes=(0, None, None, None))
        state, hist = vmap_returns(keys, rewards, context, storefn)
        return state, hist