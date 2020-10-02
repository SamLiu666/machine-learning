def say_hi():
    print('Hi, this is mymodule speaking.')


__version__ = '0.1'


if __name__ == '__main__':
    print('This program is being run by itself')
else:
    print('I am being imported from another module')
