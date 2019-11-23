
a = 5
b = 2

class MyOperations:
    
    def __init__(self,props):
        self.msg = props
    
    msg='this are my custom operations'
    
    def test(self):
        print('test',self.msg)
    
    def swap(x,y):
        return y,x

a,b = MyOperations.swap(5,2)
a = MyOperations('1')
b = MyOperations('2')