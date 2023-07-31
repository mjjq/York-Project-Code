
from datetime import datetime

from matplotlib import pyplot as plt

def savefig(name: str, **kwargs):
    date_time = datetime.now().strftime("%d-%m-%Y_%H:%M")

    s = f"./output/{date_time}_{name}.png"
    print(f"Saving figure: {s}")
    plt.savefig(s, dpi=300)

if __name__=='__main__':
    savefig("this_is_a_test")
