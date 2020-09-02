#file with useful functions

def flex_tuple(x):
    "if a tuple, returns first element; otherwise just returns"
    if isinstance(x, tuple):
        return x[0]
    else:
        return x
    
def convert_to_relu(x):
    "converts from True and False to the appropriate slopes for leaky ReLU"
    if x:
        return 1.0
    else:
        return 0.1
   
