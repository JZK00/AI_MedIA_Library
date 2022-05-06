"""
self 代表的是类的实例，代表当前对象的地址，而 self.class 则指向类。
"""


class Test:
    def prt(self):
        print(self)
        print(self.__class__)


t = Test()
t.prt()
