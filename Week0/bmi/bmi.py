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

    vals = np.array([])  
    with open("Summary.txt", "r") as f:
        for line in f.readlines():
            vals = np.append(vals, [float(line)])
    mean = np.mean(vals)
    stddev = np.std(vals)
    print(f"mean={mean:.2f} stddev={stddev:.2f}")

if __name__ == "__main__":
    main()


    