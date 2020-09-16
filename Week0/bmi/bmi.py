import numpy as np


def calc_bmi(weight, height):
    return float(weight / (height ** 2))

def read_input():
    weight = input("Enter weight in kg: ")
    if not weight:
        return 0
    height = input("Enter weight in m: ")
    if not height:
        return 0
    return calc_bmi(float(weight), float(height))

def main():
    bmi = read_input()    
    while bmi:
        with open("Summary.txt", "a") as f:
            f.writelines(str(bmi) + "\n")
        bmi = read_input()

if __name__ == "__main__":
    main()


    