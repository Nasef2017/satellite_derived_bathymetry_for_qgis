# infrastructure/canvas.py
import os
try:
    from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer, QgsProcessingContext
except ImportError:
    QgsProject = None
    QgsRasterLayer = None
    QgsVectorLayer = None
    QgsProcessingContext = None

try:
    from .raster_io import StylePostProcessor, write_qml_style
except (ImportError, ValueError):
    from infrastructure.raster_io import StylePostProcessor, write_qml_style


def add_raster_to_canvas(path, layer_name, context=None, style_path=None):
    if not path or not os.path.exists(path) or QgsProject is None:
        return

    if not style_path:
        qml = os.path.splitext(path)[0] + ".qml"
        if not os.path.exists(qml):
            qml = write_qml_style(path)
        if qml and os.path.exists(qml):
            style_path = qml

    if context:
        details = QgsProcessingContext.LayerDetails(
            layer_name, QgsProject.instance(), layer_name
        )
        if style_path and os.path.exists(style_path) and StylePostProcessor is not None:
            processor = StylePostProcessor(style_path)
            details.setPostProcessor(processor)
        context.addLayerToLoadOnCompletion(path, details)
    else:
        layer = QgsRasterLayer(path, layer_name)
        if style_path and os.path.exists(style_path):
            layer.loadNamedStyle(style_path)
            layer.triggerRepaint()
        QgsProject.instance().addMapLayer(layer)


def add_vector_to_canvas(path, layer_name, provider="ogr", context=None, style_path=None):
    if not path or not os.path.exists(path):
        return

    if not style_path:
        qml = os.path.splitext(path)[0] + ".qml"
        if os.path.exists(qml):
            style_path = qml

    if context:
        details = QgsProcessingContext.LayerDetails(
            layer_name, QgsProject.instance(), layer_name
        )
        if style_path and os.path.exists(style_path) and StylePostProcessor is not None:
            processor = StylePostProcessor(style_path)
            details.setPostProcessor(processor)
        context.addLayerToLoadOnCompletion(path, details)
    else:
        layer = QgsVectorLayer(path, layer_name, provider)
        if style_path and os.path.exists(style_path):
            layer.loadNamedStyle(style_path)
            layer.triggerRepaint()
        QgsProject.instance().addMapLayer(layer)

