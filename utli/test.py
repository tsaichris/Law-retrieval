

class testMain():
    
    par1 = True

    def __init__(self):
        self.par2 = True

    def p(self):
        print(self.par1, self.par2)

class sub(testMain):
    
    def __init__(self):
        super().__init__()

        self.par1 = False
        self.par2 = False
    def p_2(self):
        print(self.par1, self.par2)


a = sub()
a.p_2()
b = testMain()

b.p()
        