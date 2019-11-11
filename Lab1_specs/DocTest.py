# Made By Jian Gao 2019/03/08

'''
>>> x = sub.find_root(f, fprime)
>>> print(x)
7.792741452820329
>>> print(f(x))
0.0
>>> x = sub.find_root(f, fprime, MAX_ITER = 5)
>>> print(x)
7.792741452820569
>>> print(f(x))
7.318590178329032e-13
>>> x = sub.find_root(f, fprime, EPSILON = 1E-1)
>>> print(x)
7.7927448262150705
>>> print(f(x))
1.0299624985776745e-05
>>> x = sub.find_root(ff, ffprime)
>>> print(x)
1.0
>>> print(ff(x))
0.0
>>> x = sub.find_root(ff, ffprime)
>>> print(x)
1.0
>>> print(ff(x))
0.0
>>> x = sub.find_root(fff, fffprime)
>>> print(x)
4.30970363562988e-17
>>> print(fff(x))
0.0
>>> str_tree = '1 [2 [3 4       5          ] \
   6 [7 8 [9]   10 [11 12] ] \
   13\
  ]'
>>> tt = sub.make_tree(str_to_tokens(str_tree))
>>> print_tree(tt)
 1
     2
         3
         4
         5
     6
         7
         8
             9
         10
             11
             12
     13
>>> sub.max_depth(tt)
4
>>> str_tree = '1 [2 3 4 5 6 7 8]'
>>> tt = sub.make_tree(str_to_tokens(str_tree))
>>> print_tree(tt)
 1
     2
     3
     4
     5
     6
     7
     8
>>> sub.max_depth(tt)
2
>>> str_tree = '1'
>>> tt = sub.make_tree(str_to_tokens(str_tree))
>>> print_tree(tt)
 1
>>> sub.max_depth(tt)
1
>>> str_tree = '1 [2 [3 [8 9] 4 [10 11]] 5 [6 [12 13] 7 [14 15]]]'
>>> tt = sub.make_tree(str_to_tokens(str_tree))
>>> print_tree(tt)
 1
     2
         3
             8
             9
         4
             10
             11
     5
         6
             12
             13
         7
             14
             15
>>> sub.max_depth(tt)
4
>>> str_tree = '1 [2 [3 [8 [10 [6 [13 [14]]]]]]]'
>>> tt = sub.make_tree(str_to_tokens(str_tree))
>>> print_tree(tt)
 1
     2
         3
             8
                 10
                     6
                         13
                             14
>>> sub.max_depth(tt)
8
>>> str_tree = '1 [2 [3 [8 [10 [6 [13 [14 15]]]]]]]'
>>> tt = sub.make_tree(str_to_tokens(str_tree))
>>> print_tree(tt)
 1
     2
         3
             8
                 10
                     6
                         13
                             14
                             15
>>> sub.max_depth(tt)
8
>>> str_tree = '1 [2]'
>>> tt = sub.make_tree(str_to_tokens(str_tree))
>>> print_tree(tt)
 1
     2
>>> sub.max_depth(tt)
2
'''

from difflib import ndiff
import submission as sub
import math

def compare_two_files(file_1, file_2):
    diff = ndiff(open(file_1).readlines(), open(file_2).readlines())
    diff = [l for l in diff if l.startswith('+ ') or l.startswith('- ')]
    if len(diff):
        print(''.join(diff))

def test_q1():
    for i in range(0, 5000):
        if sub.nsqrt(i) != int(i ** 0.5):
            print('--------\nwrong nsqrt i ==', i)
            print('you output:', sub.nsqrt(i))
            print('expected output', int(i ** 0.5), '\n--------')
def f(x):
    return x * math.log(x) - 16.0

def fprime(x):
    return 1.0 + math.log(x)

def ff(x):
    return x ** 2 - 2 * x + 1

def ffprime(x):
    return x - 2

def fff(x):
    return (math.e ** (x) - math.e ** (-x))/(math.e ** (x) + math.e ** (-x))

def fffprime(x):
    return 1 - fff(x) ** 2

def print_tree(root, indent=0):
    print(' ' * indent, root)
    if len(root.children) > 0:
        for child in root.children:
            print_tree(child, indent+4)

import re

def myfind(s, char):
    pos = s.find(char)
    if pos == -1: # not found
        return len(s) + 1
    else: 
        return pos

def next_tok(s): # returns tok, rest_s
    if s == '': 
        return (None, None)
    # normal cases
    poss = [myfind(s, ' '), myfind(s, '['), myfind(s, ']')]
    min_pos = min(poss)
    if poss[0] == min_pos: # separator is a space
        tok, rest_s = s[ : min_pos], s[min_pos+1 : ] # skip the space
        if tok == '': # more than 1 space
            return next_tok(rest_s)
        else:
            return (tok, rest_s)
    else: # separator is a [ or ]
        tok, rest_s = s[ : min_pos], s[min_pos : ]
        if tok == '': # the next char is [ or ]
            return (rest_s[:1], rest_s[1:])
        else:
            return (tok, rest_s)
        
def str_to_tokens(str_tree):
    # remove \n first
    str_tree = str_tree.replace('\n','')
    out = []
    
    tok, s = next_tok(str_tree)
    while tok is not None:
        out.append(tok)
        tok, s = next_tok(s)
    return out

class Tree(object):
    def __init__(self, name='ROOT', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)



if __name__ == '__main__':
    import doctest
    print('test q1.....')
    test_q1()
    doctest.testmod()
    print('Test Done! If not show "Test Failed", you pass my test')
