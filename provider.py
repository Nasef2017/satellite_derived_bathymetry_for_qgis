from qgis.core import QgsProcessingProvider, QgsProcessingAlgorithm
import os, inspect, importlib.util

ALG_DIR = os.path.join(os.path.dirname(__file__), 'algorithms')

class SdbProvider(QgsProcessingProvider):
    def __init__(self):
        super().__init__()
        self._algorithms = self._load_algorithms()

    def loadAlgorithms(self):
        for alg in self._algorithms:
            self.addAlgorithm(alg)
            
    def unload(self):
        super().unload()
        self._algorithms = []

    def id(self): return 'sdb_tools'
    def name(self): return 'SDB Tools'
    def longName(self): return self.name()

    def _load_algorithms(self):
        algorithms = []
        if not os.path.isdir(ALG_DIR): return []
        for file_name in os.listdir(ALG_DIR):
            if file_name.endswith('.py') and file_name != '__init__.py':
                file_path = os.path.join(ALG_DIR, file_name)
                module_name = f"sdb_tools.algorithms.{file_name[:-3]}"
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, QgsProcessingAlgorithm) and obj is not QgsProcessingAlgorithm:
                            algorithms.append(obj())
                except Exception as e:
                    print(f"SDB Tools: Error loading {file_name}: {e}")
        return algorithms