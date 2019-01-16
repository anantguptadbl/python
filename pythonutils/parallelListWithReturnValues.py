import collections

from time import sleep
from Queue import Queue
from threading import Thread

from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=10)

def MyListDecorator(func):
    def MyListDecoratorFunction(self, index, *args):
        # Index Check
        if index == 0:
            raise IndexError('Indices start from 1')
        elif index > 0:
            index -= 1
        return func(self, index, *args)
    return MyListDecoratorFunction


class MyList(collections.MutableSequence):
    def __init__(self):
        self._inner_list = list()

    def __len__(self):
        return len(self._inner_list)

    def __contains__(self, value):
        return True if value in self._inner_list else False

    def __str__(self):
        return str(self._inner_list)
    
    def append(self, value):
        self.insert(len(self) + 1, value)
        
    def pop(self):
        # We will remove the first value
        self._inner_list.pop()
    
    @MyListDecorator
    def __delitem__(self, index):
        self._inner_list.__delitem__(index)

    @MyListDecorator
    def insert(self, index, value):
        self._inner_list.insert(index, value)

    @MyListDecorator
    def __setitem__(self, index, value):
        self._inner_list.__setitem__(index, value)

    @MyListDecorator
    def __getitem__(self, index):
        return self._inner_list.__getitem__(index)

    def map(self,funcPointer,*args,**kwargs):
        pool = ThreadPool(20)
        async_result = [pool.apply_async(funcPointer, (x,args,kwargs,)) for x in range(10)]
        retValues = [x.get() for x in async_result]
        return retValues

if __name__=="__main__":
    def randomFunc(i,*args,**kwargs):
        print("Starting to sleep for 5 second")
        sleep(5)
        print("Ended sleeping for 5 second")
        return(1)
        

    a=MyList()
    a.append(1)
    a.append(2)
    print(a)
    print(1 in a)
    print(4 in a)
    retValues=a.map(randomFunc)
    print(retValues)
