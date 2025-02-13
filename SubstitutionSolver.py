import argparse
import re
from typing import Optional, Pattern
import numpy as np
from collections import Counter
import random
import math
from tqdm import tqdm

class Solution:

    def __init__(self, invalid_chars: Optional[Pattern] = re.compile(r"[^a-zA-Z \"',.;-]")):

        # Define Valid Chars constant
        self.__invalid_chars = invalid_chars

        self.gothere = 0



    def decode(self, path_to_ciphertext: str, ciphertext_onto_plaintext: dict):

        # Read ciphertext

        file = open(path_to_ciphertext, "r")
        ciphertext = re.sub(self.__invalid_chars, "", file.read().lower())
        file.close()

        # Return the text translated with the key mapping

        return ciphertext.translate(str.maketrans(ciphertext_onto_plaintext))
    

    def find_key(self, path_to_ciphertext: str, path_to_plaintext: str, steps: Optional[int] = 7500):

        # Read ciphertext and plaintext

        file = open(path_to_ciphertext, "r")
        self.__CIPHERTEXT = re.sub(self.__invalid_chars, "", file.read().lower())
        file.close()

        file = open(path_to_plaintext, "r")
        self.__PLAINTEXT = re.sub(self.__invalid_chars, "", file.read().lower())
        file.close()

        # Initialize the key list through a simple individual character frequency analysis
        ciphertext_frequency_ranking = self.__get_character_frequency_order(self.__CIPHERTEXT)
        plaintext_frequency_ranking = self.__get_character_frequency_order(self.__PLAINTEXT)

        # Produce the initial key
        ciphertext_onto_plaintext = self.__initial_ciphertext_onto_plaintext(ciphertext_frequency_ranking, plaintext_frequency_ranking)

        # Get the initial plaintext bigram count. This is representative of the bigram
        # frequencies inherent to the english language itself, and is thus a constant.
        self.__PLAINTEXT_BIGRAMS = self.__count_bigram_probabilities(self.__PLAINTEXT)

        # ciphertext_onto_plaintext will then be modified using a hill climbing character bigram 
        # algorithm until a decent optima is reached

        # Initialize score to 0 (theoretical key maps to Opposite vectorized text)
        current_score = 0

        for step in tqdm(range(steps), desc="Finding key.", unit="steps"):
            # Get two random character keys in the ciphertext_onto_plaintext
            key1 = random.choice(list(ciphertext_onto_plaintext.keys()))
            key2 = random.choice(list(ciphertext_onto_plaintext.keys()))

            temp = ciphertext_onto_plaintext[key1]
            ciphertext_onto_plaintext[key1] = ciphertext_onto_plaintext[key2]
            ciphertext_onto_plaintext[key2] = temp


            # Determine if we should 
            new_score = self.__check_mapping(ciphertext_onto_plaintext=ciphertext_onto_plaintext, 
                                             ciphertext=self.__CIPHERTEXT)
            
            # Get the chance to randomly keep a swap regardless of performance
            threshold_to_keep = self.__exponential_decay(step=step, steps=steps)

            # Random chance to keep the swap even if the score is worse
            if(new_score > current_score or random.random() < threshold_to_keep):
                current_score = new_score
                continue
            # If it's a bad swap and we don't want to chance it, just swap
            # em back
            else:
                temp = ciphertext_onto_plaintext[key1]
                ciphertext_onto_plaintext[key1] = ciphertext_onto_plaintext[key2]
                ciphertext_onto_plaintext[key2] = temp

        # Sort into ascending order
        ciphertext_onto_plaintext = dict(sorted(ciphertext_onto_plaintext.items(), key = lambda x: x[1]))

        return ciphertext_onto_plaintext



    """
    A simple exponential decay scheduler for the above gradient ascent problem. This
    makes it so the chance to randomly keep a change starts high and decreases as the
    step size goes up.
    """
    def __exponential_decay(self, step, steps, p_init = 0.0015, p_final = 0.0003):
        k = math.log(p_init / p_final) / steps  # Compute decay rate
        return p_final + (p_init - p_final) * math.exp(-k * step)


    """
    Gets the frequency ranking of characters as returned from __count_character_occurrences,
    turning it into an ordered list where the first element is the most common character.

    param path_to_text: path to the text (plaintext or ciphertext) from which the character
    frequencies will be extracted. 

    return: ordered list of most to least common characters in text file.
    """
    def __get_character_frequency_order(self, text: str):
        char_counts = self.__count_character_occurrences(text)

        char_frequency_order = [k for k,v in sorted(char_counts.items(), key = lambda x: -x[1])]

        return char_frequency_order


    """
    Maps each character present in a possible key onto the number of times it occurs in
    a given text file (plaintext or ciphertext).

    param path_to_text: path to a plaintext or ciphertext text file

    return: dict() mapping each character in a possible key onto the number of occurrences
    in a given text file.
    """
    def __count_character_occurrences(self, text: str):

        # Convert to dict
        char_counts = Counter(text)

        return char_counts


    """
    Produce initial map of ciphertext onto plaintext characters by combining the frequency
    ranking from both texts.

    param ciphertext_frequency_ranking: The ordered ranking of characters in ciphertext from
    most to least common.
    param ciphertext_frequency_ranking: The ordered ranking of characters in plaintext from
    most to least common.

    return: initial/naive translation table of ciphertext characters onto plaintext using single 
    character frequencies.
    """
    def __initial_ciphertext_onto_plaintext(self, ciphertext_frequency_ranking: dict, plaintext_frequency_ranking: list):

        ciphertext_onto_plaintext = {}

        # Build a dict mapping each ciphertext character onto each 
        # plaintext character using the frequency rankings
        for index in range(len(ciphertext_frequency_ranking)):

            ciphertext_char = ciphertext_frequency_ranking[index]

            plaintext_char = plaintext_frequency_ranking[index]

            ciphertext_onto_plaintext[ciphertext_char] = plaintext_char

        return ciphertext_onto_plaintext 
    

    """
    Check the efficacy of the mapping during bigram hill climbing.

    param ciphertext_onto_plaintext: the current substitution mapping ciphertext->plaintext
    param ciphertext: the original ciphertext

    return: bigram accuracy score
    """
    def __check_mapping(self, ciphertext_onto_plaintext: dict, ciphertext: str):

        # Translate ciphertext using the current mapping/key
        translated_ciphertext = ciphertext.translate(str.maketrans(ciphertext_onto_plaintext))

        # Get occurrences of all two-character bigrams within translated ciphertext
        translated_ciphertext_bigram_probabilities = self.__count_bigram_probabilities(translated_ciphertext)

        ### Perform cosine similarity of occurrences of bigrams within ciphertext vs plaintext ###

        # Union translated ciphertext and plaintext bigrams
        translated_ciphertext_and_plaintext_bigrams = set(translated_ciphertext_bigram_probabilities) | set(self.__PLAINTEXT_BIGRAMS)
        
        # Extract probability vectors after unioning
        translated_ciphertext_bigram_probabilities_vector = np.array([translated_ciphertext_bigram_probabilities.get(bg, 0) for bg in translated_ciphertext_and_plaintext_bigrams])
        plaintext_bigrams_probabilities_vector = np.array([self.__PLAINTEXT_BIGRAMS.get(bg,0) for bg in translated_ciphertext_and_plaintext_bigrams])

        # Get cosine similarity :- dot over multiplied magnitude
        similarity_score = np.dot(plaintext_bigrams_probabilities_vector, translated_ciphertext_bigram_probabilities_vector) / (
            np.linalg.norm(plaintext_bigrams_probabilities_vector) * np.linalg.norm(translated_ciphertext_bigram_probabilities_vector) + 1e-10
        )

        return similarity_score

    def __count_bigram_probabilities(self, text: str):

        # Get counts of the unique bigrams in the list. This is significantly
        # faster than a for-loop
        #
        # Specifically, this creates a list of all of the two character
        # strings that exist in the given text
        bigram_counts = dict(Counter(text[i:i+2] for i in range(len(text)-1)))

        # Convert this to a mapping to probabilities
        total_bigrams = sum(bigram_counts.values())
        bigram_probabilities = {bg: count/total_bigrams for bg,count in bigram_counts.items()}

        return bigram_probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Solution")
    parser.add_argument("-p", "--plaintext", help="A text file from which rules will be determined for modern english language.", type=str, default="./plaintext.txt")
    parser.add_argument("-c", "--ciphertext", help="A cipher text to be translated into plain english through cryptanalysis.", type=str, default="./cipher.txt")
    parser.add_argument("-k", "--key", help="A key to try in the format \" \" ',-.;abcdefghijklmnopqrstuvwxyz \"", type=str, default=None)

    args = parser.parse_args()

    plaintext = args.plaintext
    ciphertext = args.ciphertext

    solution = Solution()

    ciphertext_onto_plaintext = {}

    if(args.key != None):
        key = args.key
        format = " \"',-.;abcdefghijklmnopqrstuvwxyz"

        ciphertext_onto_plaintext = {cipher: plain for cipher, plain in zip(key,format)}
    else:
        ciphertext_onto_plaintext = solution.find_key(path_to_ciphertext=ciphertext, path_to_plaintext=plaintext)

    print("APPROXIMATE PLAINTEXT : ")
    print(solution.decode(path_to_ciphertext=ciphertext, ciphertext_onto_plaintext=ciphertext_onto_plaintext))

    print("Ciphertext onto Plaintext mapping: ",ciphertext_onto_plaintext)

    print("Substitution Cipher \"key\":", "".join(ciphertext_onto_plaintext.keys()))
    print("Substitutes these values :", "".join(ciphertext_onto_plaintext.values()))


