#file with useful functions

def flex_tuple(x):
    "if a tuple, returns first element; otherwise just returns"
    if isinstance(x, tuple):
        return x[0]
    else:
        return x