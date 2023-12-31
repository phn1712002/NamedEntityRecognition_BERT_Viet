from Dataset.importTXT import importTXT

class DatasetNerBERT():
    def __init__(self, path='./Dataset/', encoding='utf-8'):
        self.path = path
        self.encoding = encoding
        self._train_dataset = None
        self._dev_dataset = None
        self._test_dataset = None

        
    def __call__(self):
        self._train_dataset = importTXT(path=self.path + 'raw/', encoding=self.encoding)()
        return self._train_dataset, self._dev_dataset, self._test_dataset