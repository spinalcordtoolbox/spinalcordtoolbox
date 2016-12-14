import matplotlib.pyplot as plt

def main():
    
    x = []
    y = []
    s = 300
    d = s/25
    for i in range (1,d):
        x.append(i)
        y.append(f(i,300))
        
    plt.plot(x, y)
    plt.show()
    
def f(x,s):
    return s/2 - x/5 
    
if __name__ == "__main__":
    main()