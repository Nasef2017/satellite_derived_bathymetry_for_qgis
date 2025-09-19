def classFactory(iface):
    from .sdb_tools_plugin import SdbToolsPlugin
    return SdbToolsPlugin(iface)