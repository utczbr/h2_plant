import json
"""
H2 Plant Configuration Editor - Complete Version
Includes: File/Edit/View/Validation menus, Run Simulation, All Nodes tab with working drag-drop
"""
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QMessageBox, QFileDialog, QDialog,
    QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QPushButton,
    QWidget, QListWidget, QListWidgetItem, QApplication, QTabWidget,
    QProgressDialog, QCheckBox, QDialogButtonBox, QScrollArea, QGridLayout, 
    QGroupBox, QHBoxLayout, QSplitter, QSizePolicy, QFrame, QProgressBar,
    QRadioButton, QSpinBox, QButtonGroup
)
from PySide6.QtCore import Qt, QTimer, QMimeData, QThread, Signal, QSettings, QRunnable, QThreadPool, QObject, Slot
from PySide6.QtGui import QColor, QShortcut, QKeySequence, QDrag, QAction
from NodeGraphQt import NodeGraph, PropertiesBinWidget, NodesPaletteWidget
import copy
from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pathlib import Path

# Core Managers
from h2_plant.gui.core.topology_inference import TopologyInferenceEngine, extract_nodes_edges_from_graph
from h2_plant.gui.core.graph_persistence import GraphPersistenceManager
from h2_plant.gui.core.advanced_validation import AdvancedValidator, ValidationLevel
from h2_plant.gui.core.worker import SimulationWorker
from h2_plant.gui.core.graph_adapter import GraphToConfigAdapter, GraphNode, GraphEdge, FlowType, Port

# Node imports
from h2_plant.gui.nodes.electrolysis import PEMStackNode, SOECStackNode, RectifierNode
from h2_plant.gui.nodes.reforming import ATRReactorNode, WGSReactorNode, SteamGeneratorNode
from h2_plant.gui.nodes.separation import PSAUnitNode, SeparationTankNode, CoalescerNode, KnockOutDrumNode, DeoxoReactorNode, TSAUnitNode
from h2_plant.gui.nodes.thermal import HeatExchangerNode, ChillerNode, DryCoolerNode
from h2_plant.gui.nodes.fluid import ProcessCompressorNode, RecirculationPumpNode
from h2_plant.gui.nodes.pumping import PumpNode 
from h2_plant.gui.nodes.logistics import ConsumerNode
from h2_plant.gui.nodes.resources import GridConnectionNode, WaterSupplyNode, AmbientHeatNode, NaturalGasSupplyNode
from h2_plant.gui.nodes.storage import (
    LPTankNode, HPTankNode, OxygenBufferNode,
    LPTankArrayNode, LPEnhancedTankNode,
    HPTankArrayNode, HPEnhancedTankNode
)
from h2_plant.gui.nodes.compression import FillingCompressorNode, OutgoingCompressorNode
from h2_plant.gui.nodes.logic import DemandSchedulerNode, EnergyPriceNode
from h2_plant.gui.nodes.utilities import BatteryNode
from h2_plant.gui.nodes.water import WaterPurifierNode, UltraPureWaterTankNode
from h2_plant.gui.themes.theme_manager import ThemeManager
# New nodes
from h2_plant.gui.nodes.energy_source import WindEnergySourceNode
from h2_plant.gui.nodes.economics import ArbitrageNode
from h2_plant.gui.nodes.mixing import MixerNode, WaterMixerNode
from h2_plant.gui.nodes.valve_node import ValveNode


class AllNodesListWidget(QListWidget):
    """Enhanced list widget with proper drag-and-drop for NodeGraphQt."""
    
    def __init__(self, graph, parent=None):
        super().__init__(parent)
        self.graph = graph
        self.setDragEnabled(True) 
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setDragDropMode(QListWidget.DragOnly)
        self.setAcceptDrops(False)

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if not item:
            return
            
        node_class = item.data(Qt.UserRole)
        if not node_class:
            return
            
        mimeData = QMimeData()
        
        node_identifier = node_class.__identifier__
        node_class_name = node_class.__name__
        full_node_type = f"{node_identifier}.{node_class_name}"
        
        # NodeGraphQt requires a specific URN format: "nodegraphqt::node:{identifier}"
        # and MIME type: "nodegraphqt/nodes"
        # The identifier part MUST match the factory key, which is full_node_type
        node_urn = f"nodegraphqt::node:{full_node_type}"
        
        mimeData.setData("nodegraphqt/nodes", node_urn.encode('utf-8'))
        
        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.exec_(Qt.CopyAction)


from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

# ==============================================================================
# GRAPH HIERARCHY - Extensible folder structure for graph categories
# ==============================================================================
# To add new folder groups:
# 1. Add a new key to GRAPH_HIERARCHY with a list of graph_ids from GRAPH_REGISTRY
# 2. Component-specific graphs can be added as new folders (e.g., "Compressor", "Tank")
#
# Available graph_ids are defined in plotter.py GRAPH_REGISTRY

GRAPH_HIERARCHY = {
    "Plant Overview": ["dispatch", "power_balance", "energy_pie"],
    "Production": ["h2_production", "oxygen_production", "cumulative_h2", "dispatch_curve"],
    "Economics": ["arbitrage", "price_histogram", "revenue_analysis"],
    "Resource Consumption": ["water_consumption", "cumulative_energy"],
    "Efficiency": ["efficiency_curve"],
    "Physics & Degradation": ["polarization", "degradation", "compressor_ts", "module_power"],
    "Thermal & Separation": ["chiller_cooling", "coalescer_separation", "kod_separation"],
}


# ==============================================================================
# PERFORMANCE OPTIMIZATION - Caching and Lazy Loading
# ==============================================================================

import hashlib
from functools import lru_cache


class FigureCache:
    """
    LRU cache for generated matplotlib figures.
    
    Caches figures by (graph_id, data_hash) to avoid regeneration when:
    - User toggles checkbox off/on
    - User switches between tabs and returns
    
    The cache is invalidated when simulation data changes (new hash).
    """
    
    def __init__(self, max_size: int = 20):
        self._cache = {}  # (graph_id, data_hash) -> Figure
        self._access_order = []  # LRU tracking
        self._max_size = max_size
        self._current_data_hash = None
    
    def get_data_hash(self, simulation_data: dict) -> str:
        """Generate a hash from simulation data for cache invalidation."""
        # Use first and last values + length as a quick fingerprint
        try:
            keys = list(simulation_data.keys())[:5]
            sample = str([(k, len(simulation_data.get(k, []))) for k in keys])
            return hashlib.md5(sample.encode()).hexdigest()[:8]
        except:
            return "unknown"
    
    def set_data(self, simulation_data: dict):
        """Update the current data hash, clearing cache if data changed."""
        new_hash = self.get_data_hash(simulation_data)
        if new_hash != self._current_data_hash:
            self.clear()
            self._current_data_hash = new_hash
    
    def get(self, graph_id: str) -> object:
        """Get cached figure or None."""
        key = (graph_id, self._current_data_hash)
        if key in self._cache:
            # Update LRU order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, graph_id: str, figure: object):
        """Cache a figure."""
        key = (graph_id, self._current_data_hash)
        
        # Evict LRU if at capacity
        while len(self._cache) >= self._max_size and self._access_order:
            old_key = self._access_order.pop(0)
            old_fig = self._cache.pop(old_key, None)
            if old_fig:
                try:
                    old_fig.clear()
                    import matplotlib.pyplot as plt
                    plt.close(old_fig)
                except:
                    pass
        
        self._cache[key] = figure
        self._access_order.append(key)
    
    def clear(self):
        """Clear all cached figures."""
        for fig in self._cache.values():
            try:
                fig.clear()
                import matplotlib.pyplot as plt
                plt.close(fig)
            except:
                pass
        self._cache.clear()
        self._access_order.clear()


class GraphWorkerSignals(QObject):
    """Signals emitted by graph generation workers."""
    graph_ready = Signal(str, object)  # graph_id, Figure or path
    error = Signal(str, str)  # graph_id, error_message
    progress = Signal(int, int, str)  # current, total, graph_name
    all_complete = Signal(dict)  # graph_id -> file_path


class ImageGenerationWorker(QThread):
    """
    Worker thread that generates all graphs as PNG files.
    
    This is the preferred approach - generates high-quality images once,
    then displays them using fast QLabel/QPixmap widgets.
    """
    progress = Signal(int, int, str)  # current, total, graph_name
    finished_with_paths = Signal(dict)  # graph_id -> file_path
    error = Signal(str)
    
    def __init__(self, simulation_data: dict, output_dir: str, graph_ids: list = None):
        super().__init__()
        self.simulation_data = simulation_data
        self.output_dir = output_dir
        self.graph_ids = graph_ids
        
    def run(self):
        """Generate all graphs as PNG files."""
        try:
            from h2_plant.gui.core.plotter import generate_all_graphs_to_files
            
            def on_progress(current, total, name):
                self.progress.emit(current, total, name)
            
            result = generate_all_graphs_to_files(
                self.simulation_data,
                self.output_dir,
                graph_ids=self.graph_ids,
                dpi=100,  # High quality
                progress_callback=on_progress
            )
            
            self.finished_with_paths.emit(result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))



class LazyGraphSlot(QFrame):
    """
    Lazy-loading placeholder that triggers graph generation when visible.
    
    Uses visibility detection to only generate graphs that are actually
    in the scroll viewport, dramatically reducing initial load time.
    """
    
    # Signal to request graph generation
    request_generation = Signal(str)  # graph_id
    
    def __init__(self, graph_id: str, graph_name: str, parent=None):
        super().__init__(parent)
        self.graph_id = graph_id
        self.graph_name = graph_name
        self._generation_requested = False
        self._is_loaded = False
        
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setMinimumHeight(200)
        self.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
            }
        """)
        
        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignCenter)
        
        # Graph name
        self._name_label = QLabel(graph_name)
        self._name_label.setStyleSheet("color: #888; font-size: 13px;")
        self._name_label.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._name_label)
        
        # Status label
        self._status_label = QLabel("Scroll to load")
        self._status_label.setStyleSheet("color: #555; font-size: 11px;")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._status_label)
    
    def check_visibility(self, viewport_rect):
        """Check if this slot is visible in the viewport and request generation."""
        if self._generation_requested or self._is_loaded:
            return
        
        # Get global position of this widget
        my_rect = self.rect()
        my_global = self.mapToGlobal(my_rect.topLeft())
        
        # Safety check: skip if widget hasn't been laid out yet (position 0,0 with small size)
        if my_global.x() == 0 and my_global.y() == 0 and my_rect.height() < 50:
            return  # Widget not ready yet
        
        my_global_rect = my_rect.translated(my_global.x(), my_global.y())
        
        # Check intersection with viewport
        if viewport_rect.intersects(my_global_rect):
            self._request_graph()
    
    def _request_graph(self):
        """Request graph generation."""
        if self._generation_requested:
            return
        self._generation_requested = True
        self._status_label.setText("Loading...")
        self._status_label.setStyleSheet("color: #2196F3; font-size: 11px;")
        self.request_generation.emit(self.graph_id)
    
    def mark_loaded(self):
        """Mark this slot as having its graph loaded."""
        self._is_loaded = True


class SimulationReportWidget(QWidget):
    """
    Widget to display simulation reports using pre-generated static images.
    
    Features:
    - PRE-GENERATION: All graphs generated as PNG files before display
    - FAST SCROLLING: Uses QLabel/QPixmap for GPU-accelerated display
    - Progress indication during image generation
    
    EXTENSIBILITY:
    - Add new graphs by registering in plotter.py GRAPH_REGISTRY
    - Add new folders by extending GRAPH_HIERARCHY above
    """
    
    # Graph display constants
    GRAPH_MIN_HEIGHT = 400
    SIDEBAR_MIN_WIDTH = 180
    SIDEBAR_DEFAULT_WIDTH = 250
    ZOOM_LEVELS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # Available zoom levels
    DEFAULT_ZOOM_INDEX = 2  # 1.0 = 100%
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_labels = {}  # graph_id -> QLabel
        self.image_paths = {}  # graph_id -> file path
        self.simulation_data = None
        self.no_data_label = None
        self._tree_items = {}
        self._generation_worker = None
        self._zoom_index = self.DEFAULT_ZOOM_INDEX  # Current zoom level index
        
        # Temp directory for generated images
        import tempfile
        self._temp_dir = tempfile.mkdtemp(prefix="h2_graphs_")
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup widget UI with QSplitter layout."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.splitter = QSplitter(Qt.Horizontal)
        
        # LEFT PANE: Sidebar
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        
        title_label = QLabel("Select Graphs")
        title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        sidebar_layout.addWidget(title_label)
        
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("All")
        deselect_all_btn = QPushButton("None")
        refresh_btn = QPushButton("Refresh Graphs")
        refresh_btn.setToolTip("Generate selected graphs")
        refresh_btn.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white;")
        
        select_all_btn.clicked.connect(self.select_all_graphs)
        deselect_all_btn.clicked.connect(self.deselect_all_graphs)
        refresh_btn.clicked.connect(self._force_refresh)
        
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        button_layout.addWidget(refresh_btn)
        sidebar_layout.addLayout(button_layout)
        
        # Zoom label (zoom via scroll or keyboard only)
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        
        self._zoom_label = QLabel("100%")
        self._zoom_label.setAlignment(Qt.AlignCenter)
        self._zoom_label.setToolTip("Use CTRL+wheel or +/- keys to zoom")
        
        zoom_layout.addWidget(self._zoom_label)
        zoom_layout.addStretch()
        sidebar_layout.addLayout(zoom_layout)
        
        self.graph_tree = QTreeWidget()
        self.graph_tree.setHeaderHidden(True)
        self.graph_tree.setIndentation(15)
        self.graph_tree.itemChanged.connect(self._on_tree_item_changed)
        self._populate_tree()
        
        sidebar_layout.addWidget(self.graph_tree, 1)
        sidebar_widget.setMinimumWidth(self.SIDEBAR_MIN_WIDTH)
        
        # RIGHT PANE: Graph display
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.graphs_container = QWidget()
        self.graphs_layout = QVBoxLayout(self.graphs_container)
        self.graphs_layout.setSpacing(12)
        self.graphs_layout.setContentsMargins(8, 8, 8, 8)
        
        self.scroll_area.setWidget(self.graphs_container)
        
        # Install event filter to intercept wheel events on scroll area
        self.scroll_area.viewport().installEventFilter(self)
        
        self.no_data_label = QLabel("No graphs available. Run simulation first.")
        self.no_data_label.setAlignment(Qt.AlignCenter)
        self.no_data_label.setStyleSheet("color: gray; font-size: 14px; padding: 50px;")
        self.graphs_layout.addWidget(self.no_data_label)
        self.graphs_layout.addStretch()
        
        self.splitter.addWidget(sidebar_widget)
        self.splitter.addWidget(self.scroll_area)
        self.splitter.setSizes([self.SIDEBAR_DEFAULT_WIDTH, 800])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(self.splitter)
    
    def _populate_tree(self):
        """Populate tree with folder hierarchy."""
        from h2_plant.gui.core.plotter import GRAPH_REGISTRY
        
        self.graph_tree.blockSignals(True)
        try:
            self.graph_tree.clear()
            self._tree_items = {}
            
            for folder_name, graph_ids in GRAPH_HIERARCHY.items():
                folder_item = QTreeWidgetItem([folder_name])
                folder_item.setFlags(folder_item.flags() | Qt.ItemIsUserCheckable)
                
                # Default behavior: Only "Plant Overview" is checked
                is_default = (folder_name == "Plant Overview")
                folder_state = Qt.Checked if is_default else Qt.Unchecked
                folder_item.setCheckState(0, folder_state)
                
                for graph_id in graph_ids:
                    if graph_id in GRAPH_REGISTRY:
                        graph_info = GRAPH_REGISTRY[graph_id]
                        child_item = QTreeWidgetItem([graph_info['name']])
                        child_item.setFlags(child_item.flags() | Qt.ItemIsUserCheckable)
                        child_item.setCheckState(0, folder_state)
                        child_item.setData(0, Qt.UserRole, graph_id)
                        child_item.setToolTip(0, graph_info.get('description', ''))
                        folder_item.addChild(child_item)
                        self._tree_items[graph_id] = child_item
                
                self.graph_tree.addTopLevelItem(folder_item)
                folder_item.setExpanded(True)
        finally:
            self.graph_tree.blockSignals(False)
    
    def _on_tree_item_changed(self, item, column):
        """Handle checkbox changes."""
        self.graph_tree.blockSignals(True)
        try:
            if item.childCount() > 0:
                new_state = item.checkState(column)
                for i in range(item.childCount()):
                    item.child(i).setCheckState(0, new_state)
            else:
                parent = item.parent()
                if parent:
                    self._update_parent_check_state(parent)
        finally:
            self.graph_tree.blockSignals(False)
        
        self._display_selected_graphs()
    
    def _update_parent_check_state(self, parent):
        """Update parent check state."""
        checked = sum(1 for i in range(parent.childCount()) 
                      if parent.child(i).checkState(0) == Qt.Checked)
        total = parent.childCount()
        
        if checked == 0:
            parent.setCheckState(0, Qt.Unchecked)
        elif checked == total:
            parent.setCheckState(0, Qt.Checked)
        else:
            parent.setCheckState(0, Qt.PartiallyChecked)
    
    def select_all_graphs(self):
        self.graph_tree.blockSignals(True)
        try:
            for i in range(self.graph_tree.topLevelItemCount()):
                folder = self.graph_tree.topLevelItem(i)
                folder.setCheckState(0, Qt.Checked)
                for j in range(folder.childCount()):
                    folder.child(j).setCheckState(0, Qt.Checked)
        finally:
            self.graph_tree.blockSignals(False)
        self._display_selected_graphs()
    
    def deselect_all_graphs(self):
        self.graph_tree.blockSignals(True)
        try:
            for i in range(self.graph_tree.topLevelItemCount()):
                folder = self.graph_tree.topLevelItem(i)
                folder.setCheckState(0, Qt.Unchecked)
                for j in range(folder.childCount()):
                    folder.child(j).setCheckState(0, Qt.Unchecked)
        finally:
            self.graph_tree.blockSignals(False)
        self._display_selected_graphs()
    
    def set_simulation_data(self, history):
        """Set simulation history data and generate all graph images."""
        self.simulation_data = history
        self._generate_all_graphs()
    
    def _get_checked_graph_ids(self):
        return [gid for gid, item in self._tree_items.items() 
                if item.checkState(0) == Qt.Checked]
    
    def load_graphs(self):
        """Reload visible graphs based on current selection."""
        self._display_selected_graphs()
    
    def _force_refresh(self):
        """Force regenerate all graphs."""
        self._generate_all_graphs()
    
    def _clear_layout(self):
        """Clear all displayed images."""
        for label in self.image_labels.values():
            label.setParent(None)
            label.deleteLater()
        self.image_labels.clear()
        
        if self.no_data_label:
            self.no_data_label.setParent(None)
            self.no_data_label.deleteLater()
            self.no_data_label = None
        
        while self.graphs_layout.count():
            item = self.graphs_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()
    
    def _generate_all_graphs(self):
        """Generate all checked graphs as PNG files in background."""
        if self.simulation_data is None:
            self._clear_layout()
            self._show_message("No simulation data. Run simulation first.", "gray")
            return
        
        checked_ids = self._get_checked_graph_ids()
        if not checked_ids:
            self._clear_layout()
            self._show_message("No graphs selected.", "gray")
            return
        
        # Show progress bar
        self._clear_layout()
        self._progress_label = QLabel("Generating graphs...")
        self._progress_label.setStyleSheet("color: #888; font-size: 14px;")
        self._progress_label.setAlignment(Qt.AlignCenter)
        self.graphs_layout.addWidget(self._progress_label)
        
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximum(len(checked_ids))
        self._progress_bar.setValue(0)
        self.graphs_layout.addWidget(self._progress_bar)
        self.graphs_layout.addStretch()
        
        # Start background worker
        self._generation_worker = ImageGenerationWorker(
            self.simulation_data, 
            self._temp_dir,
            graph_ids=checked_ids
        )
        self._generation_worker.progress.connect(self._on_generation_progress)
        self._generation_worker.finished_with_paths.connect(self._on_generation_complete)
        self._generation_worker.error.connect(self._on_generation_error)
        self._generation_worker.start()
    
    @Slot(int, int, str)
    def _on_generation_progress(self, current, total, name):
        """Update progress bar during generation."""
        if hasattr(self, '_progress_bar'):
            self._progress_bar.setValue(current)
        if hasattr(self, '_progress_label'):
            self._progress_label.setText(f"Generating: {name} ({current}/{total})")
    
    @Slot(dict)
    def _on_generation_complete(self, paths: dict):
        """Handle completion of all graph generation."""
        self.image_paths = paths
        self._display_selected_graphs()
    
    @Slot(str)
    def _on_generation_error(self, error_msg):
        """Handle generation error."""
        self._clear_layout()
        self._show_message(f"Error generating graphs: {error_msg}", "red")
    
    def _display_selected_graphs(self):
        """Display all generated images for selected graphs."""
        from PySide6.QtGui import QPixmap
        from h2_plant.gui.core.plotter import GRAPH_REGISTRY
        
        self._clear_layout()
        
        checked_ids = self._get_checked_graph_ids()
        
        if not checked_ids:
            self._show_message("No graphs selected.", "gray")
            return
        
        if not self.image_paths:
            self._show_message("No images available. Run simulation first.", "gray")
            return
        
        for graph_id in checked_ids:
            # Create container frame
            frame = QFrame()
            frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
            frame.setStyleSheet("""
                QFrame {
                    background-color: #2a2a2a;
                    border: 1px solid #3a3a3a;
                    border-radius: 6px;
                    padding: 5px;
                }
            """)
            frame_layout = QVBoxLayout(frame)
            
            # Graph title
            graph_info = GRAPH_REGISTRY.get(graph_id, {})
            title = QLabel(graph_info.get('name', graph_id))
            title.setStyleSheet("font-weight: bold; font-size: 14px; color: #eee;")
            title.setAlignment(Qt.AlignCenter)
            frame_layout.addWidget(title)
            
            filepath = self.image_paths.get(graph_id)
            if not filepath:
                # Placeholder for not generated yet
                placeholder = QLabel("Not generated yet.\nClick 'Refresh Graphs' to generate.")
                placeholder.setStyleSheet("color: #888; font-style: italic; padding: 20px;")
                placeholder.setAlignment(Qt.AlignCenter)
                placeholder.setMinimumHeight(200)
                frame_layout.addWidget(placeholder)
                self.graphs_layout.addWidget(frame)
                continue
            
            # Image display with zoom support
            image_label = QLabel()
            pixmap = QPixmap(filepath)
            if not pixmap.isNull():
                # Calculate target width with zoom
                zoom_factor = self.ZOOM_LEVELS[self._zoom_index]
                base_width = self.scroll_area.viewport().width() - 40
                target_width = int(base_width * zoom_factor)
                
                # Scale image
                scaled = pixmap.scaledToWidth(target_width, Qt.SmoothTransformation)
                image_label.setPixmap(scaled)
            else:
                image_label.setText("Failed to load image")
                image_label.setStyleSheet("color: red;")
            
            image_label.setAlignment(Qt.AlignCenter)
            frame_layout.addWidget(image_label)
            
            self.image_labels[graph_id] = image_label
            self.graphs_layout.addWidget(frame)
        
        self.graphs_layout.addStretch()
    
    def _show_message(self, text, color):
        self.no_data_label = QLabel(text)
        self.no_data_label.setAlignment(Qt.AlignCenter)
        self.no_data_label.setStyleSheet(f"color: {color}; font-size: 14px; padding: 50px;")
        self.graphs_layout.addWidget(self.no_data_label)
        self.graphs_layout.addStretch()
    
    def _zoom_in(self):
        """Increase zoom level."""
        if self._zoom_index < len(self.ZOOM_LEVELS) - 1:
            self._zoom_index += 1
            self._update_zoom_label()
            self._display_selected_graphs()
    
    def _zoom_out(self):
        """Decrease zoom level."""
        if self._zoom_index > 0:
            self._zoom_index -= 1
            self._update_zoom_label()
            self._display_selected_graphs()
    
    def _update_zoom_label(self):
        """Update the zoom level label."""
        zoom_percent = int(self.ZOOM_LEVELS[self._zoom_index] * 100)
        self._zoom_label.setText(f"{zoom_percent}%")
    
    def eventFilter(self, obj, event):
        """Filter events to prevent scrolling when CTRL is held."""
        from PySide6.QtCore import QEvent
        
        if obj == self.scroll_area.viewport() and event.type() == QEvent.Wheel:
            if event.modifiers() == Qt.ControlModifier:
                # Handle zoom directly and block scrolling
                if event.angleDelta().y() > 0:
                    self._zoom_in()
                elif event.angleDelta().y() < 0:
                    self._zoom_out()
                return True  # Block the event from reaching the scroll area
        
        return super().eventFilter(obj, event)
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        if event.modifiers() == Qt.ControlModifier:
            # Zoom with CTRL + mouse wheel (forward = zoom in, backward = zoom out)
            if event.angleDelta().y() > 0:
                self._zoom_in()
            elif event.angleDelta().y() < 0:
                self._zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for zooming."""
        if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self._zoom_in()
            event.accept()
        elif event.key() == Qt.Key_Minus or event.key() == Qt.Key_Underscore:
            self._zoom_out()
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def cleanup(self):
        """Clean up temp directory."""
        import shutil
        try:
            shutil.rmtree(self._temp_dir)
        except:
            pass



class SimulationConfigDialog(QDialog):
    """Dialog to configure simulation parameters before running."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulation Configuration")
        self.setModal(True)
        self.setMinimumWidth(300)
        
        self.selected_hours = 8760  # Default to 1 year
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Duration Selection
        group = QGroupBox("Simulation Duration")
        group_layout = QVBoxLayout(group)
        
        self.btn_group = QButtonGroup(self)
        
        self.radio_day = QRadioButton("Day (24 hours)")
        self.radio_week = QRadioButton("Week (168 hours)")
        self.radio_month = QRadioButton("Month (720 hours)")
        self.radio_year = QRadioButton("Year (8760 hours)")
        self.radio_custom = QRadioButton("Custom")
        
        self.btn_group.addButton(self.radio_day, 24)
        self.btn_group.addButton(self.radio_week, 168)
        self.btn_group.addButton(self.radio_month, 720)
        self.btn_group.addButton(self.radio_year, 8760)
        self.btn_group.addButton(self.radio_custom, 0)
        
        self.radio_year.setChecked(True)
        
        group_layout.addWidget(self.radio_day)
        group_layout.addWidget(self.radio_week)
        group_layout.addWidget(self.radio_month)
        group_layout.addWidget(self.radio_year)
        group_layout.addWidget(self.radio_custom)
        
        # Custom input
        custom_layout = QHBoxLayout()
        custom_layout.setContentsMargins(20, 0, 0, 0)
        
        self.custom_spin = QSpinBox()
        self.custom_spin.setRange(1, 100000)
        self.custom_spin.setValue(24)
        self.custom_spin.setSuffix(" hours")
        self.custom_spin.setEnabled(False)
        
        custom_layout.addWidget(QLabel("Duration:"))
        custom_layout.addWidget(self.custom_spin)
        group_layout.addLayout(custom_layout)
        
        layout.addWidget(group)
        
        # Connect signals
        self.radio_custom.toggled.connect(self._toggle_custom)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        # Change OK button text to "Run"
        buttons.button(QDialogButtonBox.Ok).setText("Run Simulation")
        
        layout.addWidget(buttons)
        
    def _toggle_custom(self, checked):
        self.custom_spin.setEnabled(checked)
        
    def get_duration_hours(self):
        if self.radio_custom.isChecked():
            return self.custom_spin.value()
        return self.btn_group.checkedId()


class PlantEditorWindow(QMainWindow):
    """H2 Plant Configuration Editor - Full Version."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("H2 Plant Configuration Editor")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create node graph
        self.graph = NodeGraph()
        
        # Initialize Managers
        self.topology_engine = TopologyInferenceEngine()
        self.persistence_mgr = GraphPersistenceManager(backup_dir=Path("./backups"))
        self.validator = AdvancedValidator()
        
        # Validation Timer
        self.validation_timer = QTimer()
        self.validation_timer.timeout.connect(self.run_validation_silent)
        
        # Set central widget with Tabs
        self.central_tabs = QTabWidget()
        
        # Tab 1: Graph Editor
        self.central_tabs.addTab(self.graph.widget, "Run Simulation")
        
        # Tab 2: Simulation Report
        self.report_widget = SimulationReportWidget()
        self.central_tabs.addTab(self.report_widget, "Simulation Report")
        
        self.setCentralWidget(self.central_tabs)
        
        # Register nodes FIRST
        self.register_all_nodes()
        
        # Setup UI components
        self.setup_docks()
        self.setup_menus()
        self.setup_toolbar()
        self.setup_context_menu()
        self.setup_keyboard_shortcuts()
        
        # Apply theme
        ThemeManager.apply_theme(self, QApplication.instance(), "dark")
        self.show()
    
    def register_all_nodes(self):
        """Register all node types."""
        self.node_classes = [
            # Electrolysis
            PEMStackNode, SOECStackNode, # RectifierNode (Implicit),
            # Storage
            # Storage
            LPTankNode, HPTankNode,
            LPTankArrayNode, LPEnhancedTankNode,
            HPTankArrayNode, HPEnhancedTankNode,
            # Compression
            FillingCompressorNode, OutgoingCompressorNode,
            # Fluid Machinery
            PumpNode, # RecirculationPumpNode (Use PumpNode), ProcessCompressorNode (Use specific compressors),
            # Sources (NEW)
            WindEnergySourceNode,
            # Economics (NEW)
            ArbitrageNode,
            # Flow Control (NEW)
            ValveNode, MixerNode, WaterMixerNode,
            # Thermal (NEW)
            ChillerNode, DryCoolerNode,
            # Separation (NEW)
            CoalescerNode, KnockOutDrumNode, PSAUnitNode, DeoxoReactorNode, TSAUnitNode,
            # Logic
            DemandSchedulerNode, EnergyPriceNode,
            # Logistics
            ConsumerNode,
            # Resources
            GridConnectionNode, WaterSupplyNode, # AmbientHeatNode, NaturalGasSupplyNode
            # Other (Disabled - Future Phase)
            # ATRReactorNode, SeparationTankNode,
            # BatteryNode, WaterPurifierNode, UltraPureWaterTankNode
        ]
        self.graph.register_nodes(self.node_classes)
    
    def setup_docks(self):
        """Setup dock widgets."""
        # Properties dock (Right)
        self.properties_bin = PropertiesBinWidget(node_graph=self.graph)
        self.prop_dock = QDockWidget("Properties", self)
        self.prop_dock.setWidget(self.properties_bin)
        self.addDockWidget(Qt.RightDockWidgetArea, self.prop_dock)

        # Nodes palette dock
        self.nodes_palette = NodesPaletteWidget(node_graph=self.graph)
        self.all_nodes_list = AllNodesListWidget(self.graph)
        
        # Populate "All Nodes" tab
        for cls in self.node_classes:
            item = QListWidgetItem(cls.NODE_NAME)
            item.setData(Qt.UserRole, cls)
            item.setToolTip(f"Drag to canvas to create {cls.NODE_NAME}")
            self.all_nodes_list.addItem(item)

        self.palette_tabs = QTabWidget()
        self.palette_tabs.addTab(self.nodes_palette, "Categories")
        self.palette_tabs.addTab(self.all_nodes_list, "All Nodes")
        
        self.palette_dock = QDockWidget("Nodes", self)
        self.palette_dock.setWidget(self.palette_tabs)
        
        # [FIX] Reverted to LeftDockWidgetArea (Standard layout)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.palette_dock)
        
        # Configure palette styling for proper node text display
        self.palette_dock.setMinimumWidth(220)  # Ensure sufficient width for node names
        self.nodes_palette.setMinimumWidth(200)
        
        # Apply styling to ensure node items are properly sized
        self.nodes_palette.setStyleSheet("""
            QTreeView::item {
                min-height: 24px;
                padding: 2px 4px;
            }
            QTreeView {
                font-size: 12px;
            }
        """)
        
        # Update palette
        self.nodes_palette.update()
    
    def setup_menus(self):
        """Setup menu bar with File, Edit, View, Validation."""
        menubar = self.menuBar()
        
        # FILE MENU
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Layout", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_layout)
        file_menu.addAction(new_action)
        
        open_action = QAction("Open Layout...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_layout)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Layout", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_layout)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save Layout As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_layout_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # EDIT MENU
        edit_menu = menubar.addMenu("Edit")
        
        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(lambda: self.graph.undo())
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(lambda: self.graph.redo())
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        delete_action = QAction("Delete", self)
        delete_action.setShortcut("Del")
        delete_action.triggered.connect(self.delete_selection)
        edit_menu.addAction(delete_action)
        
        duplicate_action = QAction("Duplicate", self)
        duplicate_action.setShortcut("Ctrl+D")
        duplicate_action.triggered.connect(self.duplicate_selection)
        edit_menu.addAction(duplicate_action)
        
        edit_menu.addSeparator()
        
        select_all_action = QAction("Select All", self)
        select_all_action.setShortcut("Ctrl+A")
        select_all_action.triggered.connect(lambda: self.graph.select_all())
        edit_menu.addAction(select_all_action)
        
        clear_selection_action = QAction("Clear Selection", self)
        clear_selection_action.setShortcut("Ctrl+Shift+A")
        clear_selection_action.triggered.connect(lambda: self.graph.clear_selection())
        edit_menu.addAction(clear_selection_action)
        
        # VIEW MENU
        view_menu = menubar.addMenu("View")
        
        fit_action = QAction("Fit to Selection", self)
        fit_action.setShortcut("F")
        fit_action.triggered.connect(lambda: self.graph.fit_to_selection())
        view_menu.addAction(fit_action)
        
        reset_zoom_action = QAction("Reset Zoom", self)
        reset_zoom_action.setShortcut("H")
        reset_zoom_action.triggered.connect(lambda: self.graph.reset_zoom())
        view_menu.addAction(reset_zoom_action)
        
        view_menu.addSeparator()
        
        toggle_props_action = QAction("Toggle Properties Panel", self)
        toggle_props_action.triggered.connect(lambda: self.prop_dock.setVisible(not self.prop_dock.isVisible()))
        view_menu.addAction(toggle_props_action)
        
        toggle_palette_action = QAction("Toggle Nodes Panel", self)
        toggle_palette_action.triggered.connect(lambda: self.palette_dock.setVisible(not self.palette_dock.isVisible()))
        view_menu.addAction(toggle_palette_action)

        # VALIDATION MENU
        validation_menu = menubar.addMenu("Validation")
        
        validate_action = QAction("Run Validation", self)
        validate_action.setShortcut("Ctrl+V")
        validate_action.triggered.connect(self.run_validation)
        validation_menu.addAction(validate_action)
        
        validation_menu.addSeparator()
        
        auto_validate_action = QAction("Auto-Validate (every 2s)", self)
        auto_validate_action.setCheckable(True)
        auto_validate_action.toggled.connect(self.toggle_auto_validation)
        validation_menu.addAction(auto_validate_action)

        # --- NEW LOCATION FOR RUN SIMULATION ---
        # Added directly to menubar to appear right of "Validation"
        run_action = QAction("Run Simulation", self)
        run_action.setShortcut("Ctrl+R")
        run_action.triggered.connect(self.run_simulation)
        menubar.addAction(run_action)

    def setup_toolbar(self):
        """Toolbar removed as requested."""
        pass
    
    def setup_context_menu(self):
        """Setup context menus."""
        try:
            graph_menu = self.graph.get_context_menu('graph')
            graph_menu.add_command('Delete', self.delete_selection)
            graph_menu.add_command('Duplicate', self.duplicate_selection)
        except Exception as e:
            print(f"Warning: Could not setup context menu: {e}")
    
    def setup_keyboard_shortcuts(self):
        """Setup additional keyboard shortcuts."""
        QShortcut(QKeySequence(Qt.CTRL | Qt.Key_D), self, self.duplicate_selection)
    
    # ---- FILE OPERATIONS ----
    def new_layout(self):
        """Create a new empty layout."""
        reply = QMessageBox.question(self, "New Layout", 
                                     "Clear current layout?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.graph.clear_session()
    
    def save_layout(self):
        """Save current layout."""
        if not hasattr(self, 'current_file') or not self.current_file:
            self.save_layout_as()
        else:
            try:
                snapshot = self.persistence_mgr.create_snapshot(self.graph, {})
                self.persistence_mgr.save(self.current_file, snapshot)
                QMessageBox.information(self, "Success", "Layout saved!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed: {e}")
    
    def save_layout_as(self):
        """Save layout with file dialog."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Layout", "", "H2_Plant Files (*.h2plant)"
        )
        if filepath:
            try:
                snapshot = self.persistence_mgr.create_snapshot(self.graph, {})
                self.persistence_mgr.save(filepath, snapshot)
                self.current_file = filepath
                QMessageBox.information(self, "Success", "Layout saved!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed: {e}")
    
    def load_layout(self):
        """Load layout from file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Layout", "", "H2_Plant Files (*.h2plant)"
        )
        if filepath:
            try:
                self.graph.clear_session()
                
                # DEBUG: Check registered nodes
                print(f"DEBUG: Registered nodes: {self.graph.registered_nodes()}")
                
                # Fix: load() returns snapshot, doesn't take graph
                snapshot = self.persistence_mgr.load(filepath)
                # Restore snapshot to graph
                self.persistence_mgr.restore_to_graph(self.graph, snapshot)
                
                self.current_file = filepath
                QMessageBox.information(self, "Success", "Layout loaded!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Load failed: {e}")
    
    # ---- EDIT OPERATIONS ----
    def delete_selection(self):
        """Delete selected nodes."""
        selected = self.graph.selected_nodes()
        if not selected:
            return
        for node in list(selected):
            try:
                self.graph.delete_node(node)
            except Exception as e:
                print(f"Error deleting node: {e}")
    
    def duplicate_selection(self):
        """Duplicate selected nodes."""
        selected = self.graph.selected_nodes()
        if not selected:
            return
        
        offset = 50
        new_nodes = []
        
        for node in selected:
            try:
                new_node = self.graph.create_node(
                    node.__class__,
                    name=f"{node.name()}_copy",
                    pos=[node.x_pos() + offset, node.y_pos() + offset]
                )
                for prop_name, prop_value in node.properties.items():
                    try:
                        new_node.set_property(prop_name, copy.deepcopy(prop_value))
                    except:
                        pass
                new_nodes.append(new_node)
            except Exception as e:
                print(f"Error duplicating node: {e}")
        
        self.graph.clear_selection()
        for node in new_nodes:
            node.set_selected(True)
    
    # ---- VALIDATION ----
    def run_validation(self):
        """Run validation and show results."""
        try:
            report = self.validator.validate(self.graph)
            if report.is_valid:
                QMessageBox.information(self, "Validation", "✓ Graph is valid!")
            else:
                issues_text = "\n".join([f"• {i.message}" for i in report.issues[:10]])
                if report.total_issues > 10:
                    issues_text += f"\n... and {report.total_issues - 10} more"
                QMessageBox.warning(self, "Validation Issues", issues_text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Validation failed: {e}")
    
    def run_validation_silent(self):
        """Run validation without popup."""
        try:
            report = self.validator.validate(self.graph)
            # Update status bar or indicator
            if hasattr(self, 'statusBar'):
                status = "✓ Valid" if report.is_valid else f"⚠ {report.total_issues} issues"
                self.statusBar().showMessage(status)
        except Exception as e:
            print(f"Validation error: {e}")
    
    def toggle_auto_validation(self, enabled):
        """Toggle automatic validation."""
        if enabled:
            self.validation_timer.start(2000)
        else:
            self.validation_timer.stop()
    
    # ---- SIMULATION ----
    def run_simulation(self):
        """Run plant simulation using new backend architecture."""
        try:
             # 0. Configure Simulation
            dialog = SimulationConfigDialog(self)
            if dialog.exec() != QDialog.Accepted:
                return
            
            duration_hours = dialog.get_duration_hours()

            # 1. Create Adapter
            adapter = GraphToConfigAdapter()
            
            # 2. Extract Nodes
            for node in self.graph.all_nodes():
                # Extract ports
                ports = []
                for p in node.input_ports():
                    ports.append(Port(name=p.name(), flow_type=FlowType.HYDROGEN, direction="input")) # Defaulting to H2 for now
                for p in node.output_ports():
                    ports.append(Port(name=p.name(), flow_type=FlowType.HYDROGEN, direction="output"))

                # Create GraphNode
                # Use get_properties() from ConfigurableNode for clean extraction
                node_props = node.get_properties() if hasattr(node, 'get_properties') else node.properties()
                
                graph_node = GraphNode(
                    id=node.id,
                    type=node.type_,
                    display_name=node.name(),
                    x=node.x_pos(),
                    y=node.y_pos(),
                    properties=node_props,
                    ports=ports
                )
                adapter.add_node(graph_node)
                
            # 3. Extract Edges
            for node in self.graph.all_nodes():
                for output_port in node.output_ports():
                    for target_port in output_port.connected_ports():
                        target_node = target_port.node()
                        
                        edge = GraphEdge(
                            source_node_id=node.id,
                            source_port=output_port.name(),
                            target_node_id=target_node.id,
                            target_port=target_port.name(),
                            flow_type=FlowType.HYDROGEN # Default, inference engine can refine this
                        )
                        adapter.add_edge(edge)
            
            # 4. Generate Context
            context = adapter.to_simulation_context()
            # Update duration from dialog
            context.simulation.duration_hours = duration_hours
            
            # 5. Create Worker
            self.worker = SimulationWorker(context)
            
            # 6. Setup Progress Dialog
            progress = QProgressDialog("Running Simulation... Please wait.", None, 0, 0, self)
            progress.setWindowTitle("Simulation in Progress")
            progress.setWindowModality(Qt.WindowModal)
            progress.setCancelButton(None)
            progress.setMinimumDuration(0)
            
            def on_finished(history):
                progress.accept()
                
                # Pass simulation data directly to report widget (no disk I/O)
                if hasattr(self, 'report_widget'):
                    self.report_widget.set_simulation_data(history)
                    self.central_tabs.setCurrentIndex(1)
                
                QMessageBox.information(self, "Simulation", "Simulation completed successfully!")
                self.worker = None
                
            def on_error(err_msg):
                progress.reject()
                QMessageBox.critical(self, "Simulation Error", f"Failed to run simulation: {err_msg}")
                self.worker = None
                
            self.worker.finished.connect(on_finished)
            self.worker.error.connect(on_error)
            
            self.worker.start()
            progress.exec_()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Simulation Error", f"Failed to start simulation: {e}")
