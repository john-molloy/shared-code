# -*- coding: utf-8 -*-
"""

REPLACE THE EXISTING PRINT.PY FILE IN FOLDER LESSON_11_PRINT


Printing

Python accepts both ' and " as string delimiters.
So which one should you use?
A common convention is this:
    Use " for normal text - things that you read.
    Use ' for string-based keys and identifiers.

"""

#%% Basic printing.

print(1, 2, 3)
print(1, 2, 3, sep=', ')
print(str(1) + ', ' + str(2))
print("Hello")


#%% Inserting values into a string for printing.
# Three methods are shown below.
#
# These methods are demonstrated using the print() function,
# but really these are methods for creating strings.  They
# are not specifically for printing.
#


#%% Method 1 - This style is still common but on the way out.

print("str = %s" % "hello")
print("str = %s" % ("again"))

# The 'd' means it's an integer.
print("a=%d" % 4)
print("a=%d, b=%d" % (4, 5))

# The 'f' means it's a float.
# The '.3' means 3 decimal places.
print("a=%d, b=%.3f" % (4, 5.6789))


# You can create the strings separately.
s = "a=%d, b=%d, c=%d" % (2, 3, 4)
print(s)
s = "a=%d, b=%d, c=%d"
print(s % (5, 6, 7))


#%% Method 2 - quite common.
# Note the 'f' prefix on the string.

n = 10
print(f"The value of n is {n}")
print(f"abc {3 * 2} def")


#%% Method 3 - This is the preferred method.
# Items in the format() function are inserted into
# the string in place of the {}.
# Indexing starts at 0, like much of Python.

"a={0}, b={1}".format(1, 2)

print("a={0}, b={1}".format(1, 2))
print("b={1}, a={0}".format(1, 2))
print("b={1}, b={1}, a={0}".format(1, 2))   # Use any order, and you can repeat.
print("a={}, b={}".format(1, 2))            # Automatic numbering
print("Hello {0}".format("John"))

print("a={0:d}, a={0:4d}, a={0:04d}".format(1))
print("a={:d}, b={:4d}".format(2, 3))
print("pi={0:.0f}, pi={0:.1f}, pi={0:.2f}, pi={0:.3f}".format(3.14159))

print("{0:<20s} {1:6.2f}".format("Spam & Eggs:", 6.99))
print("{0:>20s} {1:6.2f}".format("Spam & Eggs:", 6.99))

print("The value is {:,}".format(123456))
print("The value is {:,d}".format(123456))

print("{0:<05d}".format(999))    # 99900
print("{0:>05d}".format(999))    # 00999


#%% It doesn't have to be single values in the format() function.
# For example, you can use a dictionary.
# (Note use of the single quote ' as we're using
# them as keys into the dictionary.

let = {'a' : 'aa', 'b' : 'bb', 'c' : 'cc'}

print("Letters:")
for c in let:
    print("{single}: {double}".format(single=c, double=let[c]))


#%% More justification.

print("[{:<11}]".format("Hello"))
print("[{:^11}]".format("Hello"))
print("[{:>11}]".format("Hello"))

print("[{:.<11}]".format("Hello"))
print("[{:.^11}]".format("Hello"))
print("[{:.>11}]".format("Hello"))

print("[{:x<11}]".format("Hello"))
print("[{:x^11}]".format("Hello"))
print("[{:x>11}]".format("Hello"))


#%% And a blank string with noting added.

print("".format())


#%%
# End Of File
