


## Alpha Zero + Grammar, initial materials from Peter Graf

Here is some code to look at, and hopefully run.  Minimal summary of each file:

`alphazero_grammar.py`

This the main file that has the alpha zero implementation. It was adapted from a post by Thomas Moreland about “alpha zero in 200 lines” or something (good post, look it up).  I have hacked on this code a lot, but the core approach is his.  The goal is that the grammar stuff is unknown to this code; it should treat generating sentences from a grammar as just another “game”.  There are some exceptions in this code, but ideally it is “game-agnostic”.  I have (this morning) verified (and fixed a few things related to versions in the process) that I can at least run the code using three different games: bitstring game in “just play” mode, bitstring game in “learn the program” mode (uses the grammar stuff), and symbolic regression game.  To switch games, you can​ use command line flags, but I tend to just uncomment a couple lines in this file; you should see 3 obvious two line blocks starting at line 768.  There are other hyper- and other parameters you can fiddle with like the number of bits in the bitstring (this is called “nsites”), etc.

`netgame.py`

This file implements the “bitstring game” we’ve talked about a bit.  Pretty self-explanatory

`grammargame.py`

This file implements the abstract version of treating sentence generation from CFG as a “game”, including some masking of possible actions (you can’t fire every production from every state), etc.

`netgrammargame.py`

This is the specific implementation of the grammar stuff for the bitstring game

`srgame.py`

This is a basic attempt to use our approach for symbolic regression. On a good day this has been able to find and fit quadratic data.  Note there is no just play mode for symbolic regression.

`transformer_encoder.py`

This is the transformer encoder that takes partially built sentences as input and embeds them in a latent space from which the policy and value functions that alpha zero uses are built. It is used for the modes that use the grammar.

`helpers.py`

 These are some utility functions that the original code used; not sure how many of them are still used, but, I guess, at least one!


