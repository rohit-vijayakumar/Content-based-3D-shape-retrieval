class Dog():
    def __init__(self,name):
        self.name = name

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self,name):
        self._name = name

spud = Dog("spud")
print(spud.name)