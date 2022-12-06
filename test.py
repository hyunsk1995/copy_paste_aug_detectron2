import numpy as np
import matplotlib
matplotlib.use('Agg')
import pickle
def main():
    cooccurence_table = dict()

    with open('cooccurence_table.pkl', 'rb') as f:
        cooccurence_table = pickle.load(f)
    print_cooccurence_table(cooccurence_table)
    return

def print_cooccurence_table(table):
    for key, value in table.items():
        print(key, ":")
        for k, v in value.items():
            print("   ", k, ":", v)
    return


if __name__ == "__main__":

    main()