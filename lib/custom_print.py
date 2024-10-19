import math

class my_print():
    def __init__(self):
        self.rank=0
        self.space=''
        self.size=100
    def seperator(self):
        s='-'
        s=s*self.size
        print(s)
    def title(self,msg):
        times = math.floor((self.size - len(msg)) / 2)
        s = ' '
        s = s * times
        self.seperator()
        print(s,msg,s)
        self.seperator()
        self.rank=1
    def subtitle(self,msg):
        times=math.floor( (self.size-len(msg))/2)
        s = '-'
        s = s * times
        print(s,' ',msg,' ',s)
        self.rank=1
    def print(self,msg):
        print(self.rank*'   ',msg)
    def step_f(self,msg):
        self.rank+=1
        self.rank=max(self.rank,0)
        self.print(msg)
    def step_b(self,msg):
        self.rank -= 1
        self.print(msg)
    def step(self,x):
        self.rank += 1
        self.rank = max(self.rank, 0)
    def empty(self):
        print()
    def end(self):
        self.rank=0