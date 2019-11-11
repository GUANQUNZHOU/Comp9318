## import modules here 

################# Question 0 #################

def add(a, b): # do not change the heading of the function
    return a + b


################# Question 1 #################

def nsqrt(x): # do not change the heading of the function
    if x <= 1:
        return x
    if x == 2:
        return 1
    smaller = 0
    larger = x
    while smaller < larger:
        test = (smaller+larger)//2
        #print('la',larger)
        #print('sm',smaller)
        #print(test)
        if test**2 > x:
            larger = test
        elif test**2 < x and (test+1)**2 < x:
            smaller = test
        elif test**2 <= x and (test+1)**2 > x:
            return test
        elif (test+1)**2 == x:
            return (test+1)


################# Question 2 #################


# x_0: initial guess
# EPSILON: stop when abs(x - x_new) < EPSILON
# MAX_ITER: maximum number of iterations

## NOTE: you must use the default values of the above parameters, do not change them

def find_root(f, fprime, x_0=1.0, EPSILON = 1E-7, MAX_ITER = 1000):# do not change the heading of the function
    t = 0
    x = x_0
    while t< MAX_ITER:
        t += 1
        xj = x - f(x)/fprime(x)
        if abs(xj - x) <= EPSILON:
            return xj
        else:
            x = xj
    return xj
################# Question 3 #################

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

def make_tree(tokens): # do not change the heading of the function
    inital = Tree(tokens[0])
    parent = inital
    children = inital
    parent_node = []
    i = 1
    while i < len(tokens):
        if tokens[i] == '[':
            parent_node.append(parent)
            parent = children
            i += 1
            continue

        elif tokens[i] == ']':
            parent = parent_node.pop()
            i += 1
            continue
        else:
            children = Tree(tokens[i])
            parent.add_child(children)
            i += 1
    return parent     

def max_depth(root): # do not change the heading of the function
    if root.children == []:
        return 1
    else:
        length  = [(max_depth(de)+1) for de in root.children]
        #print(length)
        height = max(length)
        return height


