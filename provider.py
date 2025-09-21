from qgis.core import QgsProcessingProvider, QgsProcessingAlgorithm
import os
import inspect
import importlib.util
import sys

class SdbProvider(QgsProcessingProvider):

    def __init__(self):
        """Constructor."""
        super().__init__()
        self._algorithms = self._load_algorithms()

    def loadAlgorithms(self, *args, **kwargs):
        """Loads all processing algorithms from this provider."""
        for alg in self._algorithms:
            self.addAlgorithm(alg)
            
    def unload(self):
        """Unloads the provider."""
        super().unload()
        self._algorithms = []

    def id(self):
        """Returns the unique provider id."""
        return 'sdb_tools'

    def name(self):
        """Returns the provider name."""
        return 'SDB Tools'

    def longName(self):
        """Returns the a more verbose provider name."""
        return self.name()

    def _load_algorithms(self):
        """Dynamically loads all algorithm classes from .py files."""
        algorithms = []
        alg_folder = os.path.join(os.path.dirname(__file__), 'algorithms')
        if not os.path.isdir(alg_folder):
            return []

        for f in os.scandir(alg_folder):
            if f.is_file() and f.name.endswith('.py') and f.name != '__init__.py':
                module_name = f'sdb_tools.algorithms.{f.name[:-3]}'
                
                # In case the plugin was reloaded, remove the old module
                if module_name in sys.modules:
                    del sys.modules[module_name]

                try:
                    spec = importlib.util.spec_from_file_location(module_name, f.path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, QgsProcessingAlgorithm) and obj is not QgsProcessingAlgorithm:
                            algorithms.append(obj())
                except Exception as e:
                    print(f"SDB Tools: Error loading algorithm from {f.name}: {e}")
        return algorithms
