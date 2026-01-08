# symbolic regression with AlphaGrammar

import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import numpy as np
import random
from copy import deepcopy

import matplotlib.pyplot as plt

from nltk.parse.recursivedescent import RecursiveDescentParser
from nltk.tokenize import wordpunct_tokenize

from scipy.optimize import least_squares
from sympy.parsing import sympy_parser
import sympy
import lmfit

from .grammargame import GrammarAgent, GrammarEnv


operators = ['/', '*', '+', '-', '**']
functions = ['cos', 'sin']

def prefixToInfix(prefix):
    prefix = deepcopy(prefix)
    stack = []
    prefix = prefix[::-1]
    for element in prefix:
        if element in operators:
            first = stack.pop()
            second = stack.pop()
            tempResult = "(" + first+element+second + ")"
            stack.append(tempResult)
        elif element in functions:
            first = stack.pop()
            tempResult =  element +  "(" +first + ")"
            stack.append(tempResult)
        elif element != "":
            stack.append(element)
    return stack.pop()

def prefixStringToInfix(prefix):
    form = prefix.strip().split(" ")
    return prefixToInfix(form)


class SRGameEnv():
    def __init__(self, grammar_env):
        # set up a sillly regression problem to test on
        my_grammar_env = deepcopy(grammar_env)
        self.grammar_agent = GrammarAgent(deepcopy(self), my_grammar_env)
        self.Cmin = 1.0
        self.Cmax = 4.0
        self.Xmin = 1
        self.Xmax = 3
        nx = 41
        xmin = self.Xmin
        xmax = self.Xmax
        self.xs = np.linspace(xmin, xmax,nx)

        self.generate_new_function()
        self.reset()

    def prefix_str_to_py_fn(self, fn_form):
        param_values = {}
        ind_vars = ()
        if 'x'  in fn_form:
            param_values['x'] = self.xs
            ind_vars = ('x',)
        fn_form = fn_form.strip().split(" ")
        for term in fn_form:
            if "C" in term:
                param_values[term] = 0.5 * (self.Cmin + self.Cmax)
        fn_form = prefixToInfix(fn_form)
        #print ("FF", fn_form, ind_vars)
        sympy_model = sympy_parser.parse_expr(fn_form, evaluate=False)
        sympy_model = sympy.lambdify(list(sympy_model.free_symbols), sympy_model)
        return sympy_model, param_values, ind_vars

    def fit_core(self, fn_form, plot=True):
        # fit fn_form to data using leastsquares
        sympy_model, param_values, ind_vars = self.prefix_str_to_py_fn(fn_form)
        lm_mod = lmfit.Model(sympy_model, independent_vars=ind_vars)
        res = lm_mod.fit(data=self.exact_ys, **param_values)
        if plot:
            print ("FIT done", res)
            res.plot_fit()
            plt.plot(self.xs, self.exact_ys, label='true')
            plt.legend()
            plt.show()
        return res
    
    def fit(self, fn_form, plot=True):
        res = self.fit_core(fn_form, plot)
#        r2 = res.rsquared
        if res.success:
            ss_res = np.sum((res.data - res.best_fit)**2)
            ss_tot = np.sum((res.data - np.mean(res.data))**2)
            r2 = 1 - (ss_res / ss_tot)
        p = 1
        sgn =  1 if r2 >= 0 else -1
        return  sgn * np.power(np.abs(r2),p)

    def generate_new_function(self):
        if True:
            fn_form = "+ * 4 sin * 4 x  +  C0  +  * C1 x   * C2 * x x"
            self.sympy_model, self.param_values, self.ind_vars = self.prefix_str_to_py_fn(fn_form)
            for term in self.param_values.keys():
                if "C" in term:
                    self.param_values[term] = self.Cmin + (self.Cmax-self.Cmin) * np.random.random()
        else:
            # this generates a whole new functional form
            # This works fine, but the functions are too easy!
            grammar = self.grammar_agent.grammar_env.grammar
            fn_form = self.grammar_agent.generate_one(grammar, grammar.start(), 50)
            fn_form = " ".join(fn_form)

            # THERE IS A BUG SOMEWHERE, BC THIS CODE CORRUPTS THE ENV
    #         env = deepcopy(self.grammar_agent.grammar_env)
    #         agent = GrammarAgent(deepcopy(self), env)
    # #        self.grammar_agent.grammar_env.reset()
    #         env.reset()
    #         done = False
    #         while not done:
    #             act = agent.random_valid_action()
    #             obs, rew, done, trunc, info = env.step_with_mask(act)
    #         fn_form = info['rule']
            self.sympy_model, self.param_values, self.ind_vars = self.prefix_str_to_py_fn(fn_form)
            for term in self.param_values.keys():
                if "C" in term:
                    self.param_values[term] = self.Cmin + (self.Cmax-self.Cmin) * np.random.random()
        ys = self.sympy_model(**self.param_values)
        print ("CREATED FUNCTION:", fn_form)
        return ys

    def generate_new_coefficients_and_data(self):
        # this just generates new coefficients for a quadratic with all 3 terms
#         a = self.Cmin
#         b = self.Cmax
#         C = (b-a)*np.random.random_sample(3)+a
# #            print ("FN RESET  ", C)
#         self.exact_fn = lambda x: C[0] + C[1]*x + C[2]*x**2
        for term in self.param_values.keys():
            if "C" in term:
                self.param_values[term] = self.Cmin + (self.Cmax-self.Cmin) * np.random.random()
        ys = self.sympy_model(**self.param_values)
        return ys

    def reset(self):
        # set up a sillly regression problem to test on
        if False:
            # fixed problem
            self.exact_fn = lambda x: 1 + 2*x + 3*x**2
            self.exact_ys = self.exact_fn(self.xs)
        else:
            # vary the problem each episode
            self.exact_ys = self.generate_new_coefficients_and_data()


class GrammarSREnv(gym.Env):
    def __init__(self):
        # self.grammarstr =  """
        # S -> E | '+' E E  | '+' '+' E E E
        # E -> '+' E E | C0' | '*' 'C1' 'x' | '*' 'C2' '*' 'x' 'x' 
        # """
        self.grammarstr =  """
        S -> '+' S S | 'C0' | '*' 'C1' 'x' | '*' 'C2' '*' 'x' 'x' | '*' S S | '/' S S | '*' C3 'sin' '*' C4 'x' 
        """
#        S -> '+' S S | 'C0' | '*' 'C1' 'x' | '*' 'C2' '*' 'x' 'x' | '*' S S | '/' S S | '*' '5' 'sin' '*' '5' 'x' | 'cos' S
        # c0 + c1 * x + c2 * x * x  == length 11
        max_len = 15
        self.grammar_env = GrammarEnv(self.grammarstr, max_len)
        self.game_env = SRGameEnv(self.grammar_env)

        self.observation_space = self.grammar_env.observation_space
        self.action_space = self.grammar_env.action_space
        ##self.agent = SRGrammarAgent(self.game_env, self.grammar_env)  no agent need for SR,
        ## because there is no underlying "game" with "actions" to generate and use.

    # These steps are unnecessary, actually, because returned rule is a prefix expression
    # that, after conversion to infix, can directly be parsed and "lambdfied" by sympy into
    # a python callable
    # def extract_parsed_rules(self, parsed):
    #     # build python code (?) from math expression generated by grammar 
    #     self.fn_form = "y = x**2"

    # def evaluate_parsed_rules(self):
    #     # fit expression to data for a regression task
    #     reward = self.game_env.fit(self.fn_form)
    #     return reward

    def step(self, action):
        state, reward, done, trunc, info = self.grammar_env.decode_and_step(action)
        if done or trunc:
            # very much grammar specific code:
            # complete rule has been built; 
            # evaluate here how well it does at SR task(s).
            reward += self.game_env.fit(info['rule'], plot=False)
 #           rule = prefixStringToInfix(info['rule'])
 #           print ("RULE CREATED  ", rule, "            ", reward)

        return state, reward, done, trunc, info

    def get_action_mask(self, state = None):
        return self.grammar_env.get_action_mask(state)
    
    def reset(self):
        self.grammar_env.reset()
        self.game_env.reset()
        return self.grammar_env.state, {}


class GrammarOnesEnv(gym.Env):
    def __init__(self):
        self.grammarstr =  """
        S -> S S | '1'
        """
        max_len = 8
        self.grammar_env = GrammarEnv(self.grammarstr, max_len)

        self.observation_space = self.grammar_env.observation_space
        self.action_space = self.grammar_env.action_space

    def step(self, action):
        state, reward, done, trunc, info = self.grammar_env.decode_and_step(action)
        if done:
            fn_form = info['rule'].strip().split(" ")
            reward += fn_form.count("1")
 #           print ("RULE CREATED  ", fn_form, "            ", reward)
        return state, reward, done, trunc, info
    
    def get_optimal_policy(self, state):
        # return discrete pdf pi corr
        # lean on option mask:  there are 2 productions, first is the one that will grow the string.
        # Optimal policy grow string until it can't anymore, then fills in the ones, or, at least,
        # never use up the last S until you have to.  mask will not allow S->S S if string is too long, so all we need is
        # to check case where there is only one S. If so, all probabillity goes on it.  Otherwise, uniform over allowed actions is fine.
        mask = self.grammar_env.get_action_mask(state)
        statelist = list(state)
        if statelist.count(0) > 1:
            pi = mask / sum(mask.flatten())
        else:
            assert statelist.count(0) == 1 # otherwise string is in done state
            idx = statelist.index(0)
            pi = np.zeros_like(mask)
            if statelist.count(2)==1:  # only 1 pad token left, can't apply S->S S anymore!
                pi[idx,1] = 1
            else:
                pi[idx,0] = 1  # force S->S S
        pi = pi.flatten()
        return pi



    def get_action_mask(self, state = None):
        return self.grammar_env.get_action_mask(state)
    
    def reset(self):
        self.grammar_env.reset()
        return self.grammar_env.state



def make_grammarsrgame():
    env = GrammarSREnv()
    return env

def make_grammaronesgame():
    env = GrammarOnesEnv()
    return env

def main():
    grammar_env = make_grammarsrgame()
    # random testing of stuff
    srenv = SRGameEnv(grammar_env)
    srenv.fit("+ C1 * C2 * x x", plot=False)


    # parser = RecursiveDescentParser(grammar_env.grammar_env.grammar)
    # #parser.trace(2)
    # rule = "*  C2  *  x  x"
    # rule = "+  C0            +     * C1 x     *  C2  *  x  x"
    # rule = "+     + C0        C0    * C1 x    "
    # parsed,  = parser.parse(wordpunct_tokenize(rule))

    agent = GrammarAgent(srenv, grammar_env.grammar_env)
    for _ in range(5):
        grammar_env.reset()
        agent.grammar_env.reset()
        done = False
        its = 1
        while not done:
    #        act = agent.random_action()
            if its < 4:
                act = [grammar_env.grammar_env.real_state_len - 1,0]
            else:
                act = agent.random_valid_action()
    #        obs, rew, done, trunc, info = grammar_env.step(act)
            obs, rew, done, trunc, info = grammar_env.grammar_env.step_with_mask(act)
            its += 1
        res = srenv.fit_core(info['rule'], plot=False)
        print ("RULE", info['rule'], "\n        rsquared",  res.rsquared)
        print ("        ", res.params)

if __name__=="__main__":
    main()
