

if __name__ == "__main__":
    x=0.0
    n=0.0005
    counter=0
    while True:
        nn=max(n,n*x)
        x=(1-nn)*x+nn
        print(x)
        counter+=1
        if x>=0.999:break
    print(counter)

