#!/usr/bin/python

class TestA(object):
    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, m):
        self._m = m

    def get_m(self):
        try:
            return self._m
        except:
            return None

A = TestA()
m = A.get_m()
print m is None

A.m = 10
m = A.get_m()
print m is None
