# Substitution Solver #
This code automatically determines the key of a ciphertext. This is done as such :

* This code performs a frequency analysis on a large amount of modern plaintext to determine the frequency at which different ascii characters in r"[a-zA-Z0-9.!?,;'" ]" appear. 
* Then, ciphertext is similarly analyzed to see the frequency at which cipher characters appear. 
* The character frequencies in the plaintext and ciphertext are then correlated to one another to determine an Approximate key automatically.

To run this code, open a terminal, and while within the project directory, first run 

`pip install -r requirements.txt` 

to install requirements and then simply run Solution.py on the command line using Python 3.10, like : 

`python3.10 Solution.py --ciphertext ./path_to_ciphertext --plaintext ./path_to_plaintext`

This will generate an Approximate key using a character frequency analysis in conjunction
with a bigram frequency analysis hill-climbing algorithm to convert the ciphertext into
text which Approximates English.

It is recommended that this project is run using pyenv for python version management, but
this is optional.

