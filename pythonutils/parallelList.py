import collections

from Queue import Queue
from threading import Thread


class Worker(Thread):
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            try:
                func(*args, **kargs)
            except Exception, e:
                print e
            finally:
                self.tasks.task_done()


class ThreadPool:
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for _ in range(num_threads):
            Worker(self.tasks)

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue"""
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        self.tasks.join()

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
        [pool.add_task(funcPointer,x,args,kwargs) for x in self._inner_list]
        pool.wait_completion()

if __name__=="__main__":
    def randomFunc(i,*args,**kwargs):
        print("Starting to sleep for 1 second")
        sleep(1)
        print("Ended sleeping for 1 second")

    a=MyList()
    a.append(1)
    a.append(2)
    print(a)
    print(1 in a)
    print(4 in a)
    a.map(randomFunc)
