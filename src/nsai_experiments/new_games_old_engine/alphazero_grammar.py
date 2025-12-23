#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-player Alpha Zero
@author: Thomas Moerland, Delft University of Technology
"""

import numpy as np
# import tensorflow as tf
# #tf.compat.v1.enable_eager_execution()
# import tensorflow.contrib.slim as slim

import torch
import random
myseed = 1
torch.manual_seed(myseed)
np.random.seed(myseed)
random.seed(myseed)


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

from transformer_encoder import EncoderOnlyTransformer

import argparse
import os
import time
import copy
from gym import wrappers
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from helpers import (argmax,check_space, store_safely,stable_normalizer,Database)
from netgame import make_netgame
from netgrammargame import make_grammarnetgame
from srgame import make_grammarsrgame, make_grammaronesgame
from grammargame import GrammarEnv

"""
For the basic bitflipping game we need a grammar for its rules.  Can imagine the env comes with a grammar, so the
search code is generic, just gets a generic grammar to work with

But starting from rules for bitflip, we can back out a grammar to play with:

for i in range(len(state)):
    if state[i] == 0, then set_state[i]=1

for_all i \in [0,len(state)], state(i,0) => set_state(i,1)

need predicates/actions;
pick-bit,  check-bit, set-bit, flip-bit, next-bit, start-bit, ...

simplest: assume 'rule applier' knows some heuristic like 'check all bits until condition is met, then apply action'
Then grammar can be:
s-> C A
C -> bit0 | bit1
A -> set0 | set 1

"""

"""
3 architectures?  many envs with different obs/action spaces.  Must enumerate to sort it out:

            obs                                       action
cartpole: Box len 4                                 discrete len 2 (push right or left)
netgame:  multibinary len nsites                    discrete nsites (fill/flip one of nsites bits)
legame:   MultiBinary len nsites in window          discrete 2 (fill middle of window or not)
srgame:                         ----- not a gym env ---
grammargame: MultiDiscrete [nsym X max sentence length]       MultiDiscrete [max sentence len X num productions], has mask
"""

#### Neural Networks ##
class Model(nn.Module):
    def __init__(self,arch, Env,lr,n_hidden_layers,n_hidden_units):
        super(Model, self).__init__()
        self.arch = arch
        self.genv = Env  if isinstance(Env, GrammarEnv) else None# only used outside __init__ if Env is a grammar environment

        self.action_dim, self.action_discrete  = check_space(Env.action_space)
        self.state_dim, self.state_discrete  = check_space(Env.observation_space)
        # if not self.action_discrete: 
        #     raise ValueError('Continuous action space not implemented')
        if arch == "mlp":
            # Check the Gym environment
            dim = np.array(self.state_dim)[0]
            self.lin1 = nn.Linear(dim, n_hidden_units)
            self.hidden_layers = nn.ModuleList()
            for _ in range(n_hidden_layers):
                self.hidden_layers.append(nn.Linear(n_hidden_units, n_hidden_units))
        elif arch == "lstm":
            self.action_dim = np.prod(Env.action_space.nvec)
            input_dim = Env.observation_space.shape[0]
            self.lstm = nn.LSTM(input_dim, n_hidden_units, num_layers=n_hidden_layers)
            # yes packing
            # self.lstm = nn.LSTM(1, n_hidden_units, num_layers=n_hidden_layers, batch_first =True)
            # The linear layer that maps from hidden state space to tag space
            self.hidden2tag = nn.Linear(n_hidden_units, n_hidden_units)
        elif arch == "transformer":
            self.action_dim = np.prod(Env.action_space.nvec)
            self.ntoks = self.genv.nsym + 1
            vocab_size = self.ntoks + 1 # add one because we will use "special token" to extract encoding for shole sequence
            d_model = n_hidden_units
            nhead = 4
            num_layers = 6
            dim_feedforward = 128
            self.encoder = EncoderOnlyTransformer(vocab_size, d_model, nhead, num_layers, dim_feedforward)
            self.hidden2tag = nn.Linear(n_hidden_units, n_hidden_units)
            self.n_hidden_units = n_hidden_units
        elif arch == "gnn":
            self.psg_net = PowerSystemsNetwork()
        
        else:
            raise ValueError('arch must be one of: mlp, lstm, transformer')

        self.linV = nn.Linear(n_hidden_units, 1)
        self.linpi = nn.Linear(n_hidden_units, self.action_dim)
        self.softmax = nn.Softmax(dim=1) #self.action_dim)
        # self.bn1 = nn.BatchNorm2d(n_hidden_units)
        # self.bn2 = nn.BatchNorm2d(n_hidden_units)

        self.activation = nn.ELU()

        self.n_hidden_layers = n_hidden_layers

        # Loss and optimizer
#        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)  
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)  
        self.clear_training_round_stats()
#        summary(self, (1,dim))

    def clear_training_round_stats(self):
        self.tstats = {'total' : 0, 'policy':0, 'value':0, 'cnt':0}
    def get_training_round_stats(self):
        return self.tstats

    def forward_lstm(self, x):
        # no packing
        x = torch.as_tensor(x, dtype=torch.float32)
        x, _ = self.lstm(x)
        x = self.hidden2tag(x)

        # trying to do packing:
        # size = x.size
        # seq_lengths = [np.argmax(x[0]==-1)]
        # if seq_lengths[0] > 1:
        #     print ("bad")
        # x = torch.as_tensor(x, dtype=torch.float32).reshape(1, size, 1)
        # pack = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
        # out, _ = self.lstm(pack)
        # unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # x = self.hidden2tag(unpacked).reshape(1,np.product(unpacked.shape))
        return x

    def forward_transformer(self, xin):
        def get_len(arr, element):
            """get length of "real" sequence"""
            index = np.where(arr == element)[0]
            if index.size > 0:
                return index[0]
            else:
                return len(arr)

        # prepend all the samples with the special token that will be used as the full sequence encoding to send to the 
        # policy and value layers
        bs = xin.shape[0]
        special_tok = self.genv.pad_tok+1
        special_col = torch.full((bs,1), special_tok)
        samples = torch.cat((special_col, torch.tensor(xin, dtype=torch.int)), dim=1)

        # build the mask
        seq_len = self.genv.state_len
        mask = torch.zeros((bs,seq_len+1))  #+1 to allow for special token we added to sequence
        for i in range(bs):
            lengths = [get_len(x, self.genv.pad_tok) for x in xin]
            mask[i,lengths[i]+1:] = True  # mask past desired truncated sequence length.

        # run the encoder, grab encoding of special token
        xout = self.encoder.forward_special_token(samples, src_key_padding_mask=mask)
        return xout
    
    def forward_gnn(self, xin):
        xout = self.psg_net(xin)
        return xout

    def forward(self, x):
        if self.arch == "mlp":
            x = torch.as_tensor(x, dtype=torch.float32)
            x = self.activation(self.lin1(x))
            for layer in self.hidden_layers:
                x = self.activation(layer(x))
        elif self.arch == "lstm":
            x = self.forward_lstm(x)
        elif self.arch == "transformer":
            x = self.forward_transformer(x)
        elif self.arch == "gnn":
            x = self.forward_gnn(x)
        return x        

    def predict_V(self,s):
        x = self.forward(s)
        self.V_hat = self.linV(x)  # value head
        return self.V_hat
        
    def predict_pi_logits(self,s):
        x = self.forward(s)
        log_pi_hat = self.linpi(x)
        self.pi_hat = log_pi_hat
        return log_pi_hat
    
    def predict_pi_logits_masked(self,s):
        # if s.shape[0] != 1:
        #     raise Exception ("Not ready for batch predictions")
        x = self.forward(s)
        log_pi_hat = self.linpi(x)
        fullmask = np.zeros_like(log_pi_hat.detach().numpy())
        for i in range(s.shape[0]):
            mask = self.genv.get_action_mask(state=s[i])
            mask = mask.flatten()
            fullmask[i,:] = mask
#        mask = torch.Tensor(mask.flatten()).type(torch.BoolTensor)
        mask = torch.Tensor(fullmask).type(torch.BoolTensor)
        log_pi_hat = torch.where(mask, log_pi_hat, torch.tensor(-1e+8))
        return log_pi_hat

    def predict_pi(self,s):
        if self.genv is not None:
            log_pi_hat = self.predict_pi_logits_masked(s)
        else:
            log_pi_hat = self.predict_pi_logits(s)
        self.pi_hat = self.softmax(log_pi_hat) # policy head           
        return self.pi_hat

    
    def train_once(self,sb,Vb,pib):
        sb = np.array(sb)
        Vb = np.array(Vb)
        pib = np.array(pib) 
        # print ("TRAINING <sb vb pib>", sb.shape, Vb.shape, pib.shape)
        # for i in range(sb.shape[0]):
        #     print (sb[i,:], Vb[i,:], pib[i,:])
    
        #self.train()
        # one epoch of training
        log_pi_hat = self.predict_pi_logits(sb)
        self.predict_V(sb)
        # forward pass
        # Loss
        lossV = nn.MSELoss()
        outputV = lossV(torch.Tensor(Vb), self.V_hat)
        losspi = nn.CrossEntropyLoss()
        outputpi = losspi(log_pi_hat, torch.Tensor(pib))
        lam = 1
        loss = outputpi + lam*outputV
#        print ("LOSSES <bs, l, lpi, lv>", len(sb), loss, outputpi, outputV)
        # backward pass
        #self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

        # for p in self.parameters():
        #     print(p.grad.norm())

        # update weights
        self.optimizer.step()

#        New average = old average * (n-1)/n + new value /n
        self.tstats['cnt'] += 1
        n = self.tstats['cnt']
        outputpi = outputpi.item()
        outputV = outputV.item()
        self.tstats['policy'] = self.tstats['policy'] * (n-1)/n + outputpi /n
        self.tstats['value'] = self.tstats['value'] * (n-1)/n + outputV /n
        self.tstats['total'] = self.tstats['policy'] + self.tstats['value']

    



##### MCTS functions #####
      
class Action():
    ''' Action object '''
    def __init__(self,index,parent_state,Q_init=0.0):
        self.index = index
        self.parent_state = parent_state
        self.W = 0.0
        self.n = 0
        self.Q = float(Q_init)
                
    def add_child_state(self,s1,r,terminal,model,Env):
        self.child_state = State(s1,r,terminal,self,self.parent_state.na,model,Env)
        return self.child_state
        
    def update(self,R):
        R = R.detach().numpy()
        self.n += 1
        self.W += R
        self.Q = float(self.W/self.n)

class State():
    ''' State object '''

    def __init__(self,index,r,terminal,parent_action,na,model,Env):
        ''' Initialize a new state '''
        self.index = copy.deepcopy(index) # state
        self.r = r # reward upon arriving in this state
        self.terminal = terminal # whether the domain terminated in this state
        self.parent_action = parent_action
        self.n = 0
        self.model = model
        
        self.evaluate()
        # Child actions
        self.na = na
        self.child_actions = [Action(a,parent_state=self,Q_init=self.V.detach().numpy()) for a in range(na)]
        self.priors = model.predict_pi(index[None,]).flatten()
        # debug code: for net game when state is C A [3 4 ...] or C A C A, does the correct pi get learned
        # if index[0] == 3:
        #     print ("PRIORS: ", index)
        #     print (self.priors.detach().numpy().reshape((6,8)))  # hard coded for state_len=6, grammar has 8 productions
        # debug code: for net game when state is C A [3 4 ...] or C A C A, does the correct pi get learned
        # if not terminal and index[0] == 0:
        #     print ("PRIORS: ", index, ", ", Env.grammar_env.decode_state(index))
        #     shape = Env.grammar_env.action_space.nvec
        #     print (self.priors.detach().numpy().reshape((shape)))  # hard coded for state_len=6, grammar has 8 productions


        # debug: if we seed MCTS with exact answer
        if False:
            correct_pi = int(Env.should_activate(Env.unflatten_subgrid(index[None,])))
            self.priors = np.zeros(2)
            self.priors[correct_pi] = 1
            self.priors = torch.Tensor(self.priors)

    
    def select(self,c=1.5):
        ''' Select one of the child actions based on UCT rule '''
        BIGM = 1e8
        priors = self.priors.detach().numpy()
#        UCT = np.array([child_action.Q + prior * c * (np.sqrt(self.n + 1)/(child_action.n + 1)) for child_action,prior in zip(self.child_actions,priors)]) 
#        UCT = np.array([child_action.Q * (prior>1e-8) + prior * c * (np.sqrt(self.n + 1)/(child_action.n + 1)) for child_action,prior in zip(self.child_actions,priors)]) 
        UCT = np.array([-BIGM * (prior<1e-8) + child_action.Q * (prior>1e-8) + prior * c * (np.sqrt(self.n + 1)/(child_action.n + 1)) for child_action,prior in zip(self.child_actions,priors)]) 
#        qpart = np.array([child_action.Q * (prior>1e-8) for child_action,prior in zip(self.child_actions,priors)]) 
#        print ("PRIORS", priors)
### This is very bad, as expected (unless priors are perfect):        UCT = np.array([prior for child_action,prior in zip(self.child_actions,priors)]) 

        winner = argmax(UCT)
        # print ("UCT", UCT, winner)
        # print ("QPART", qpart)
        # if winner > 40:
        #     print ("STOPME")
        return self.child_actions[winner]

    def evaluate(self):
        ''' Bootstrap the state value '''
#        print(self.index)
        self.V = np.squeeze(self.model.predict_V(self.index[None,])) if not self.terminal else torch.Tensor([0.0])          

    def update(self):
        ''' update count on backward pass '''
        self.n += 1
        
class MCTS():
    ''' MCTS object '''

    def __init__(self,root,root_index,model,na,gamma):
        self.root = None
        self.root_index = root_index
        self.model = model
        self.na = na
        self.gamma = gamma
    
    def search(self,n_mcts,c,Env):
        ''' Perform the MCTS search from the root '''
        if self.root is None:
            self.root = State(self.root_index,r=0.0,terminal=False,parent_action=None,na=self.na,model=self.model, Env=Env) # initialize new root
        else:
            self.root.parent_action = None # continue from current root
        if self.root.terminal:
            raise(ValueError("Can't do tree search from a terminal state"))
        
#        print ("Starting MCTS episode <s>", self.root.index)

        for i in range(n_mcts):     
            state = self.root # reset to root for new trace
            mcts_env = copy.deepcopy(Env) # copy original Env to rollout from
#            print ("(Re)Starting MCTS episode <s>", mcts_env.state, state.index)

            while not state.terminal: 
                action = state.select(c=c)
                s1,r,t,_,_ = mcts_env.step(action.index)
#                print ("MCTS step <a, s, r, t>", action.index, s1, r, t)
                if hasattr(action,'child_state'):
    # This goes by the ACTION having children, not the state; involves the question of whether we can deal with _continuous_ states where 
    # we never get the exact same state again;  this is NOT a problem; as long as the env is deterministic, the same action from the same
    # state always reaches the same next state, so indeed we CAN reuse the MCTS tree within an episode 
    # (noting that each episode we build a brand new tree, so a new initial state is not a problem)
                    state = action.child_state # select
#                    print ("existing state" , state.index)
                    # if state.terminal:
                    #     print ("CHILD STATE is terminal")
                    continue
                else:
                    state = action.add_child_state(s1,r,t,self.model,mcts_env) # expand
                    # if state.terminal:
                    #     print ("Episode is terminal")
        #This makes a new State, which uses the network to predict pi and V but then ends this iteration of the search
        # This is where we could do a "rollout" of some length using pi, and backing out the "reward + value(from the V-net)" from a deeper level
#                    print ("expand new child state", state.index)
                    break # note this ends episode.

            # Back-up 
            R = state.V         
            while state.parent_action is not None: # loop back-up until root is reached
                R = state.r + self.gamma * R 
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()                
    
    def return_results(self,temp):
        ''' Process the output at the root node '''
        counts = np.array([child_action.n for child_action in self.root.child_actions])
        Q = np.array([child_action.Q for child_action in self.root.child_actions])
        pi_target = stable_normalizer(counts,temp)
        V_target = np.sum((counts/np.sum(counts))*Q)[None]
#        print ("VTARG", V_target)
#        print ("COUNTS", counts, sum(counts))
        return self.root.index,pi_target,V_target
    
    def forward(self,a,s1):
        ''' Move the root forward '''
        if not hasattr(self.root.child_actions[a],'child_state'):
            self.root = None
            self.root_index = s1
        elif np.linalg.norm(self.root.child_actions[a].child_state.index - s1) > 0.01:
            # point here is that S1 is the result of actually taking the action in the env, whereas
            # self.root.child_actions[a].child_state.index is the state reached DURING MCTS when we tried that action
            # in the root state; they should be the same in a non-stochastic environment.  Since this (the contruction of a MCTS tree 
            # that is shared between steps) all happens DURING
            # a SINGLE episode, it doesn't matter if the env is randomly initialized in reset().
            print('Warning: this domain seems stochastic. Not re-using the subtree for next search. '+
                  'To deal with stochastic environments, implement progressive widening.')
            self.root = None
            self.root_index = s1            
        else:
            self.root = self.root.child_actions[a].child_state

    def dump_tree(self):
        def recursive_part(state, level):
            sp = ' '
            print(sp*level*3 + "state", state.index, state.n)
            for child in state.child_actions:
                print (sp*level*3 + "child action", child.index, child.n, child.W, child.Q)
                if hasattr(child,'child_state'):
                    recursive_part(child.child_state, level+1)
        
        print ("DUMP tree")
        recursive_part(self.root, 0)



def test_net_model(model, env, plot=False):
    nsamples = 100
    xs = []
    Vs = []
    Vstar = []
    ncorrect = 0
    for i in range(nsamples):
        state = env.observation_space.sample()
        teststate = state.reshape(1,len(state))
        V = model.predict_V(teststate).detach().numpy()
        pi = model.predict_pi(teststate).detach().numpy()
        correct_val = len(state) - sum(state)
        best_pi = argmax(pi)
        correct_pi = state[best_pi] == 0
        ncorrect +=  correct_pi
        if (plot):
            print (i, V[0], correct_val, V[0]-correct_val, "                ", correct_pi)
        xs.append(i)
        Vs.append(V[0])
        Vstar.append(correct_val)

    if plot:
        fig,ax = plt.subplots(1,figsize=[7,5])

        ax.scatter(xs,Vs, color='red')
        ax.scatter(xs,Vstar, color='green')

        ax.set_ylabel('Value')
        ax.set_xlabel('TestSample',color='darkred')
        plt.savefig(os.getcwd()+'/val_fn_test.png',bbox_inches="tight",dpi=300)

    Vs = np.array(Vs).reshape((nsamples))
    Vstar = np.array(Vstar).reshape(Vs.shape)
    prop_correct = ncorrect/nsamples
    val_err_vect = Vs - Vstar
    val_err = np.linalg.norm(val_err_vect)
    #mine = np.sqrt(sum([(Vs[i] - Vstar[i])**2 for i in range(nsamples)]))
    #me2 = np.sqrt(val_err_vect.dot(val_err_vect))
    #print (prop_correct, val_err, mine, Vs.shape, Vstar.shape, val_err_vect.shape)
    return prop_correct, val_err

def test_leg_model_just_sample(model, env, plot=False):
    # test by just sampling observations
    nsamples = 100
    ncorrect = 0
    nones = 0
    for i in range(nsamples):
        state = env.observation_space.sample()
        state = np.random.choice([0, 1], size=(len(state),), p=[.75, .25])        
        teststate = state.reshape(1,len(state))
        V = model.predict_V(teststate).detach().numpy()
        pi = model.predict_pi(teststate).detach().numpy()
        best_pi = argmax(pi)
        correct_pi = env.should_activate(env.unflatten_subgrid(teststate)) 
        was_correct = best_pi == correct_pi
        ncorrect +=  was_correct
        nones += correct_pi 
        if (plot):
            print (i, V[0], was_correct, "[", best_pi, correct_pi, "]", nones)
    prop_correct = ncorrect/nsamples
    #mine = np.sqrt(sum([(Vs[i] - Vstar[i])**2 for i in range(nsamples)]))
    #me2 = np.sqrt(val_err_vect.dot(val_err_vect))
    #print (prop_correct, val_err, mine, Vs.shape, Vstar.shape, val_err_vect.shape)
    return prop_correct, nones

def test_leg_model(model, env, plot=False):
    # test by running an actual episode
    ncorrect = 0
    nones = 0
    done = False
    myenv = copy.deepcopy(env)
    obs, _ = myenv.reset(plot=plot)
    steps = 0
    while not done:
        obs = obs.reshape((1,len(obs)))
        pi = model.predict_pi(obs).detach().numpy()
        best_pi = argmax(pi)
        correct_pi = env.should_activate(env.unflatten_subgrid(obs)) 
        was_correct = best_pi == correct_pi
        ncorrect +=  was_correct
        nones += correct_pi
        obs, r, done, _, _ = myenv.step(best_pi)
        steps += 1

    prop_correct = ncorrect/steps
    #mine = np.sqrt(sum([(Vs[i] - Vstar[i])**2 for i in range(nsamples)]))
    #me2 = np.sqrt(val_err_vect.dot(val_err_vect))
    #print (prop_correct, val_err, mine, Vs.shape, Vstar.shape, val_err_vect.shape)
    return prop_correct, nones

def supervised_data(env):
    state = env.observation_space.sample()
    # for legame
    # teststate = state.reshape(1,len(state))
    # correct_pi = int(env.should_activate(env.unflatten_subgrid(teststate)))
    # pi = np.zeros(2)
    # pi[correct_pi] = 1
    # others?...
    # for OnesEnv, state is list of zeros (S) and ones ("1")
    # grammar is 2 productions 0: S -> S S, 1: S->"1"
    # optimal action is to invoke 0 in some place where there's a 0 (S) unless string is too long, then start filling via S->"1",
    max_len = env.grammar_env.state_len
    trunc_to = np.random.randint(0,max_len)
    state[trunc_to:] = env.grammar_env.pad_tok
    pi = np.zeros((2,max_len))
    sidx = np.where(state == 0)[0]
    good_prod = 0 if trunc_to < max_len - 1 else 1
    if len(sidx) > 0:
        pi[good_prod,sidx] = 1 / len(sidx) 
    val = max_len - list(state).count(1)
    return state, pi.flatten(), val

#### Agent ##
def agent(args):
    game=args.game
    arch=args.arch
    n_ep=args.n_ep
    n_mcts=args.n_mcts
    max_ep_len=args.max_ep_len
    lr=args.lr
    c=args.c
    gamma=args.gamma
    data_size=args.data_size
    batch_size=args.batch_size
    temp=args.temp
    n_hidden_layers=args.n_hidden_layers
    n_hidden_units=args.n_hidden_units
    nsites = args.nsites

    ''' Outer training loop '''
    real_run = True # activate debugging only code or not

    episode_returns = [] # storage
    timepoints = []
    # Environments
    env_for_model = None
    if "grammar" in game.lower():
        if "net" in game.lower():        
            Env = make_grammarnetgame(nsites=nsites)
        elif "leg" in game.lower():
            Env = make_grammarlegame(nsites=nsites)
        elif "sr" in game.lower():
            Env = make_grammarsrgame()
        elif "ones" in game.lower():
            Env = make_grammaronesgame()
        elif "power" in game.lower():
            Env = make_grammarpowersystemgame()            
        env_for_model = Env.grammar_env
    else:
        if "net" in game.lower():
            Env = make_netgame(nsites=nsites)
        elif "leg" in game.lower():
            Env = make_legame(nsites=nsites)
        elif "power" in game.lower():
            Env = make_powersystemgame(args)
        else:
            Env = make_game(game)
        env_for_model = Env
    model = Model(arch=arch, Env=env_for_model,lr=lr,n_hidden_layers=n_hidden_layers,n_hidden_units=n_hidden_units)  

    D = Database(max_size=data_size,batch_size=batch_size)        
    t_total = 0 # total steps   
    R_best = -np.Inf
 
    for ep in range(n_ep):    
        start = time.time()
        s, _ = Env.reset() 
        R = 0.0 # Total return counter
        this_episode_returns = [] # storage for this episode
        a_store = []
        seed = myseed  #np.random.randint(1e7) # draw some Env seed
        #Env.seed(seed)    
        pi_stats = {'mcts': 0, 'net':0, 'steps':0, 'ones':0}  

        mcts = MCTS(root_index=s,root=None,model=model,na=model.action_dim,gamma=gamma) # the object responsible for MCTS searches                             
        for t in range(max_ep_len):
            if real_run:
                # MCTS step
                mcts.search(n_mcts=n_mcts,c=c,Env=Env) # perform a forward search
                state,pi,V = mcts.return_results(temp) # extract the root output
                if False:
                    # more debugging: state from mcts, but its pi from supervision:
                    # correct_pi = int(Env.should_activate(Env.unflatten_subgrid(state)))
                    # pi = np.zeros(2)
                    # pi[correct_pi] = 1
                    pi = Env.get_optimal_policy(state)
            else:
                # debugging: can we learn pi in supervised fashion?
                state,pi,V = supervised_data(Env)

            print ("Results returned", state, pi, V)
            # mcts.dump_tree()  # This is very illuminating!
            D.store((state,V,pi))

            # debug: just use results from net (not mcts), goes with case of training by supervised learning (cheat)
            if not real_run:
                state = Env.grammar_env.state
            # test: does NN predict same action?
            netpi = model.predict_pi(state.reshape(1,len(state)))
            netpi = netpi.detach().numpy()
            neta = argmax(netpi)
            mctsa = argmax(pi)

            if not real_run:
                a = neta
            else:
                a = np.random.choice(len(pi),p=pi)

            # Is mcts really improving the policy?:
            if "grammar" not in game.lower():
                if "net" in game.lower():
                    pi_stats['mcts'] += Env.state[a] == 0
                    pi_stats['net'] += Env.state[neta] == 0
                else:
                    correct_a = int(Env.should_activate(Env.unflatten_subgrid(state)))
    #            print ("COMPARE pi's <correct, mcts, net>", correct_a, a, neta)
                    pi_stats['mcts'] += correct_a == a
                    pi_stats['net'] += correct_a == neta
                    pi_stats['ones'] += correct_a

            pi_stats['steps'] += 1
  
            a_store.append(a)
            # Make the true step
            s1,r,terminal,truncated, info = Env.step(a)

            if terminal:
# If this was a grammar env, there was a rule created, then it was used to try to "do something using the rule", and that is the reward
                if "grammar" in game.lower():
                    print ("RULE CREATED  ", info['rule'], "   ", r)
# Would this be a time to cache rules that have been too often generated.  For grammars, since a "rule" == S, an S->S S production means
# we can add the cached rule as a fixed S and disable the production that produced it.  (may be a sequence of productions?)
# example: 
#         S -> '+' S S | 'C0' | '*' 'C1' 'x' | '*' 'C2' '*' 'x' 'x'
# gets stuck on c2 x^2 term.
# Oh wait! My proposed modification to the grammar results in the exact same grammar!
# Can avoid this with better grammar? -- cached expressions can't be the whole sentence
#         S -> '+' S S | 'C0' | '*' 'C1' 'x' | '*' 'C2' '*' 'x' 'x' | S + E
#         E -> 'null' or something, but we change E.
# Does this prevent S -> E, i.e. just stuck in same trap?
# Problem: If we change grammar on the fly, don't we have to start learning all over again?  How does a human (seemlessly almost) do this?


#            print ("True step <a,s,r,t,pi, neta, mctsa>", a, s1, r, terminal, pi, netpi, mctsa, neta)
            R += r
            this_episode_returns.append(r)
            t_total += n_mcts # total number of environment steps (counts the mcts steps)                

            if terminal:
                break
            else:
                if real_run:
                    mcts.forward(a,s1)
        
        # Finished episode
        episode_returns.append(R) # store the total episode return
        timepoints.append(t_total) # store the timestep count of the episode return
        store_safely(os.getcwd(),'result',{'R':episode_returns,'t':timepoints})  
#        print ("GAME OVER", episode_returns, this_episode_returns, sum(this_episode_returns))
        print ("GAME OVER", sum(this_episode_returns))

        if R > R_best:
            a_best = a_store
            seed_best = seed
            R_best = R
        
        # Train
        D.reshuffle()
        model.clear_training_round_stats()
        for epoch in range(1):
            for sb,Vb,pib in D:
                model.train_once(sb,Vb,pib)
        training_round_stats = model.get_training_round_stats()
        print ("Learning session average losses", training_round_stats['total'], training_round_stats['policy'], training_round_stats['value'])
        if game.lower()[0:3] == "net":
            pi_correct, val_err = test_net_model(model, Env, plot=False)
            print('Finished episode {}, total return: {}, total time: {} sec,  prop_pi_correct: {},   val_err: {},  mcts_good: {:.2f}, net_good: {:.2f}'.format(
                ep,np.round(R,2),np.round((time.time()-start),1), pi_correct, val_err, pi_stats['mcts']/pi_stats['steps'], pi_stats['net']/pi_stats['steps']))
        elif game.lower()[0:3] == "leg":
            pi_correct, nones = test_leg_model(model, Env, plot=False)
            goodness = Env.evaluate_goodness(Env.solution)
            print()
            lgt.display_grid(Env.get_grid())
            print('Finished episode {}, total return: {}, total time: {} sec,  prop_pi_correct: {:.2f}, goodness: {}, nones: {}, mcts_good: {:.2f}, net_good: {:.2f}, epi_ones: {}'.format(
                ep,np.round(R,2),np.round((time.time()-start),1), pi_correct, goodness, nones, pi_stats['mcts']/pi_stats['steps'], 
                pi_stats['net']/pi_stats['steps'], pi_stats['ones']), info['reason'])
        else:
            print('Finished episode {}, total return: {}, total time: {} sec'.format(
                ep,np.round(R,2),np.round((time.time()-start),1)))

    #pi_correct, val_err = test_model(model, Env, plot=True)

    # Return results
    return episode_returns, timepoints, a_best, seed_best, R_best

#### Command line call, parsing and plotting ##
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#    parser.add_argument('--game', default='CartPole-v0',help='Training environment')
    # parser.add_argument('--game', default='net',help='Training environment')
    # parser.add_argument('--arch', default='mlp',help='neural net architecture: mlp, lstm, transformer')

    # parser.add_argument('--game', default='grammarsr',help='Training environment')
    # parser.add_argument('--arch', default='transformer',help='neural net architecture: mlp, lstm, transformer')

    parser.add_argument('--game', default='grammarnet',help='Training environment')
    parser.add_argument('--arch', default='transformer',help='neural net architecture: mlp, lstm, transformer')


    parser.add_argument('--n_ep', type=int, default=3000, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=20, help='Number of MCTS traces per step')
    parser.add_argument('--max_ep_len', type=int, default=3000, help='Maximum number of steps per episode')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
#    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')  # orig
    parser.add_argument('--c', type=float, default=1.5, help='UCT constant')  # orig
#    parser.add_argument('--c', type=float, default=10, help='UCT constant')
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature in normalization of counts to policy target')
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount parameter')
    parser.add_argument('--data_size', type=int, default=1000, help='Dataset size (FIFO)')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--window', type=int, default=25, help='Smoothing window for visualization')
    parser.add_argument('--nsites', type=int, default=20, help='size of grid to play netgame on')

    parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers in NN')  
    parser.add_argument('--n_hidden_units', type=int, default=128, help='Number of units per hidden layers in NN')

    # options for Claude-generated PowerSystemGame environment.
    parser.add_argument('--agent', default='test_reward',help='test agent: greedy|random|minimal|test_flow|test_reward')
    parser.add_argument('--map', default='test',help='map to use: conuus|none|test')
    parser.add_argument('--nnodes', default=9,help='number of candidate generator nodes', type=int)
    parser.add_argument('--nloads', default=4,help='number of candidate loads', type=int)
    parser.add_argument('--rseed', default=2,help='random seed', type=int)
    parser.add_argument('--epilen', default=5,help='episode length', type=int)


    
    args = parser.parse_args()
    episode_returns,timepoints,a_best,seed_best,R_best = agent(args)

    # # Finished training: Visualize
    # fig,ax = plt.subplots(1,figsize=[7,5])
    # total_eps = len(episode_returns)
    # episode_returns = smooth(episode_returns,args.window,mode='valid') 
    # ax.plot(symmetric_remove(np.arange(total_eps),args.window-1),episode_returns,linewidth=4,color='darkred')
    # ax.set_ylabel('Return')
    # ax.set_xlabel('Episode',color='darkred')
    # plt.savefig(os.getcwd()+'/learning_curve.png',bbox_inches="tight",dpi=300)
    


#     print('Showing best episode with return {}'.format(R_best))
# #    Env = make_game(args.game)
#     Env = make_netgame()
#     Env = wrappers.Monitor(Env,os.getcwd() + '/best_episode',force=True)
#     for runs in range(1000):
#         Env.reset()
#         Env.seed(seed_best)
#         done = False
#         i = 0
#         while not done:
#             a = a_best[i]
#             _,_,done,_ = Env.step(a)
#             Env.render()
#             i += 1
