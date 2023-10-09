import sys,os

p = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

print(p)

sys.path.append(
    p
)
