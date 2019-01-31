def print_info(*args,**kwargs):
    """This is doc string"""
    for x in args:
        print(x)
    for k in kwargs.keys():
        print("{}={}".format(k,k))

print_info(*[1,2,3,4,5],**{'1':1,'2':2,'4':3})
print_info(__doc__)