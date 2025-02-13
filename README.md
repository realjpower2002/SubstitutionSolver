# Assignment 1 Solution #
This code automatically determines the key of a ciphertext. This is done as such :

* This code performs a frequency analysis on a large amount of modern plaintext to determine the frequency at which different ascii characters in r"[a-zA-Z0-9.!?,;'" ]" appear. 
* Then, ciphertext is similarly analyzed to see the frequency at which cipher characters appear. 
* The character frequencies in the plaintext and ciphertext are then correlated to one another to determine an Approximate key automatically.

To run this code, simply run Solution.py on the command line using Python 3.10 like : 

`python3.10 Solution.py --ciphertext ./path_to_ciphertext --plaintext ./path_to_plaintext`

This will generate an Approximate key using a character frequency analysis in conjunction
with a bigram frequency analysis hill-climbing algorithm to convert the ciphertext into
text which Approximates English.

