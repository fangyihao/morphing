from functools import lru_cache
from operator import itemgetter
import nltk
nltk.download('punkt')
def longest_common_substring(x: str, y: str) -> (int, int, int):
     
    # function to find the longest common substring
 
    # Memorizing with maximum size of the memory as 1
    @lru_cache(maxsize=1) 
     
    # function to find the longest common prefix
    def longest_common_prefix(i: int, j: int) -> int:
       
        if 0 <= i < len(x) and 0 <= j < len(y) and x[i] == y[j]:
            return 1 + longest_common_prefix(i + 1, j + 1)
        else:
            return 0
 
    # diagonally computing the subproblems
    # to decrease memory dependency
    def digonal_computation():
         
        # upper right triangle of the 2D array
        for k in range(len(x)):       
            yield from ((longest_common_prefix(i, j), i, j)
                        for i, j in zip(range(k, -1, -1),
                                    range(len(y) - 1, -1, -1)))
         
        # lower left triangle of the 2D array
        for k in range(len(y)):       
            yield from ((longest_common_prefix(i, j), i, j)
                        for i, j in zip(range(k, -1, -1),
                                    range(len(x) - 1, -1, -1)))
 
    # returning the maximum of all the subproblems
    return max(digonal_computation(), key=itemgetter(0), default=(0, 0, 0))

# Driver Code
if __name__ == '__main__':
    x: str = 'a man and a woman sitting next to each other'
    y: str = 'a man and woman sitting on a couch looking at a laptop computer'
    x = nltk.word_tokenize(x)
    y = nltk.word_tokenize(y)
    length, i, j = longest_common_substring(x, y)
    print(f'length: {length}, i: {i}, j: {j}')
    print(f'x substring: {x[i: i + length]}')
    print(f'y substring: {y[j: j + length]}')