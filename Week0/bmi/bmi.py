import numpy as np


def calc_bmi(weight, height):
    return float(weight / (height ** 2))

def read_input():
    weight = float(input("Enter weight in kg: "))
    if not weight:
        return 0
    height = float(input("Enter weight in m: "))
    if not height:
        return 0
    return bmi(weight, height)

def main():
    bmi = read_input()    
    while bmi:
        with open("Summary.txt", "a") as f:
            f.writelines(str(bmi))
        bmi = read_input()

if __name__ == "__main__":
    main()


    