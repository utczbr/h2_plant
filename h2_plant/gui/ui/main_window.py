import logging
import yaml
from functools import partial
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
    QRadioButton, QSpinBox, QButtonGroup, QComboBox, QLineEdit, QFormLayout,
    QAbstractItemView, QDoubleSpinBox, QPlainTextEdit, QToolButton, QStyle
)
from PySide6.QtCore import Qt, QTimer, QMimeData, QThread, Signal, Slot, QEvent
from PySide6.QtGui import QColor, QShortcut, QKeySequence, QDrag, QAction, QPalette
from NodeGraphQt import NodeGraph, PropertiesBinWidget, NodesPaletteWidget
import copy
from pathlib import Path
from typing import Any, List, Optional, Tuple
import matplotlib.pyplot as plt

# Core Managers
from h2_plant.gui.core.topology_inference import TopologyInferenceEngine, extract_nodes_edges_from_graph
from h2_plant.gui.core.graph_persistence import GraphPersistenceManager
from h2_plant.gui.core.advanced_validation import AdvancedValidator, ValidationLevel
from h2_plant.gui.core.worker import SimulationWorker
from h2_plant.gui.core.graph_adapter import GraphToConfigAdapter, GraphNode, GraphEdge, FlowType, Port
from h2_plant.gui.core.scenario_bundle_exporter import export_bundle
from h2_plant.config.models import EconomicsConfig, SimulationConfig

logger = logging.getLogger(__name__)
from h2_plant.gui.core.scenario_visual_importer import (
    ScenarioVisualImporter,
    infer_equipment_file_for_topology,
    resolve_component_id_for_equipment,
    resolve_simulation_source,
)
from h2_plant.gui.core.prebuilt_visual_layout import ensure_prebuilt_layout_file
from h2_plant.gui.core.prebuilt_visual_layout import prebuilt_layout_needs_regeneration
from h2_plant.gui.core.scenario_param_mapper import backend_to_gui_props
from h2_plant.gui.core.scenario_workspace import (
    DEFAULT_EQUIPMENT_FILE,
    DEFAULT_OPEX_FILE,
    DEFAULT_SIMULATION_FILE,
    copy_into_workspace,
    create_workspace_from_sources,
    load_yaml_preview,
    refresh_manifest_file_hashes,
    resolve_manifest_file,
)
from h2_plant.gui.core.economics_editor import (
    apply_general_econ_info,
    extract_general_econ_info,
    load_yaml_text,
    validate_capex_yaml_text,
    validate_opex_yaml_text,
)
from h2_plant.gui.core.simulation_results_loader import (
    ReportTableData,
    TablePayload,
    load_capex_data,
    load_lcoh_data,
    load_opex_data,
)

# Node imports
from h2_plant.gui.nodes.electrolysis import PEMStackNode, SOECStackNode, RectifierNode
from h2_plant.gui.nodes.separation import (
    PSAUnitNode, CoalescerNode, KnockOutDrumNode, DeoxoReactorNode,
    HydrogenMultiCycloneNode, SeparationTankNode, SyngasPSANode,
)
from h2_plant.gui.nodes.thermal import (
    ChillerNode, DryCoolerNode,
    InterchangerNode, ElectricBoilerNode, AttemperatorNode, CoolingManagerNode,
)
from h2_plant.gui.nodes.water import (
    WaterPurifierNode, UltraPureWaterTankNode,
    ExternalWaterSourceNode, WaterPumpThermodynamicNode,
)
from h2_plant.gui.themes.theme_manager import ThemeManager
from h2_plant.gui.nodes.mixing import (
    MixerNode,
    StreamSplitterNode, DrainRecorderMixerNode, SignalMakeupMixerNode,
    ProportionalMakeupMixerNode, OxygenMakeupNode,
)
from h2_plant.gui.nodes.valve_node import ValveNode
from h2_plant.gui.nodes.storage import DetailedTankNode, DischargeStationNode, CompressorSingleNode
from h2_plant.gui.nodes.reforming import IntegratedATRPlantNode, ATRBoilerNode, BiogasSourceNode
from h2_plant.gui.nodes.scenario_component import ScenarioComponentNode

PREBUILT_VISUAL_VARIANTS = {
    "SOEC + PEM + ATR": {
        "topology_file": "plant_topology.yaml",
        "equipment_file": "Economics/equipment_mappings.yaml",
        "prebuilt_filename": "plant_topology_visual.h2plant",
        "project_name": "Plant Topology Visual Twin (SOEC + PEM + ATR)",
    },
    "SOEC + PEM": {
        "topology_file": "plant_topology_PEM+SOEC.yaml",
        "equipment_file": "Economics/equipment_mappings_PEM+SOEC.yaml",
        "prebuilt_filename": "plant_topology_visual_pem_soec.h2plant",
        "project_name": "Plant Topology Visual Twin (SOEC + PEM)",
    },
}


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

def _build_graph_hierarchy():
    """Build GRAPH_HIERARCHY dynamically from catalog categories.
    
    Ensures all registered graphs are reachable from the GUI tree,
    and no stale IDs produce empty folders.
    """
    try:
        from h2_plant.visualization.graph_catalog import GRAPH_REGISTRY
        hierarchy = {}
        for category in sorted(GRAPH_REGISTRY.list_categories()):
            graphs = GRAPH_REGISTRY.get_by_category(category)
            ids = [g.graph_id for g in graphs if g.enabled]
            if ids:
                hierarchy[category] = ids
        return hierarchy
    except Exception:
        # Fallback if catalog import fails
        return {
            "Plant Overview": ["dispatch", "energy_pie"],
            "Production": ["h2_production", "oxygen_production", "cumulative_h2"],
            "Economics": ["arbitrage", "price_histogram"],
        }

GRAPH_HIERARCHY = _build_graph_hierarchy()


# ==============================================================================
# PERFORMANCE OPTIMIZATION - Caching and Lazy Loading
# ==============================================================================

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
        self._stop_requested = False
        
    def run(self):
        """Generate all graphs as PNG files."""
        try:
            from h2_plant.visualization.graph_catalog import GRAPH_REGISTRY, GraphLibrary
            from h2_plant.visualization import static_graphs
            import os
            
            result_paths = {}
            total = len(self.graph_ids)
            
            # Prepare Data - Normalize once
            norm_history = static_graphs.normalize_history(self.simulation_data)
            
            def on_progress(current, total, name):
                self.progress.emit(current, total, name)
            
            for index, graph_id in enumerate(self.graph_ids):
                if self._stop_requested:
                    break

                metadata = GRAPH_REGISTRY.get(graph_id)
                if not metadata:
                    continue
                
                # Update progress
                on_progress(index + 1, total, metadata.title)
                
                try:
                    if metadata.library == GraphLibrary.MATPLOTLIB:
                        fig = metadata.function(norm_history)
                        if fig:
                            filename = f"{graph_id}.png"
                            filepath = os.path.join(self.output_dir, filename)
                            fig.savefig(filepath, dpi=100, bbox_inches='tight')
                            plt.close(fig)
                            result_paths[graph_id] = filepath
                    elif metadata.library == GraphLibrary.PLOTLY:
                        fig = metadata.function(norm_history)
                        if fig:
                            filename = f"{graph_id}.html"
                            filepath = os.path.join(self.output_dir, filename)
                            fig.write_html(filepath, include_plotlyjs=True)
                            result_paths[graph_id] = filepath
                            
                except Exception as g_err:
                    logger.warning(f"Error generating graph {graph_id}: {g_err}")
            
            self.finished_with_paths.emit(result_paths)
            
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            self.error.emit(str(e))

    def request_stop(self):
        """Request cooperative cancellation — checked between graph renders."""
        self._stop_requested = True


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
        self._results_output_dir: Path | None = None
        self._results_scenarios_dir: Path | None = None
        self.no_data_label = None
        self._tree_items = {}
        self._generation_worker = None
        self._zoom_index = self.DEFAULT_ZOOM_INDEX  # Current zoom level index
        self._resize_refresh_active = False
        self._resize_refresh_timer = QTimer(self)
        self._resize_refresh_timer.setSingleShot(True)
        self._resize_refresh_timer.timeout.connect(self._refresh_images_after_resize)
        
        # Temp directory for generated images
        import tempfile
        self._temp_dir = tempfile.mkdtemp(prefix="h2_graphs_")
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup widget UI with report subtabs."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.results_tabs = QTabWidget()
        self.graphs_tab = self._build_graphs_tab()
        self.capex_tab = self._build_capex_tab()
        self.opex_tab = self._build_opex_tab()
        self.lcoh_tab = self._build_lcoh_tab()

        self.results_tabs.addTab(self.graphs_tab, "Graphs")
        self.results_tabs.addTab(self.capex_tab, "CAPEX")
        self.results_tabs.addTab(self.opex_tab, "OPEX")
        self.results_tabs.addTab(self.lcoh_tab, "LCOH")

        main_layout.addWidget(self.results_tabs)

    def _build_graphs_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

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
        theme = self._theme_tokens()
        refresh_btn.setStyleSheet(
            f"""
            QPushButton {{
                font-weight: bold;
                background-color: {theme["accent"]};
                color: {theme["button_text"]};
            }}
            QPushButton:hover {{
                background-color: {theme["accent_hover"]};
            }}
            """
        )

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
        self.no_data_label.setStyleSheet(
            f"color: {theme['muted_text']}; font-size: 14px; padding: 50px;"
        )
        self.graphs_layout.addWidget(self.no_data_label)
        self.graphs_layout.addStretch()

        self.splitter.addWidget(sidebar_widget)
        self.splitter.addWidget(self.scroll_area)
        self.splitter.setSizes([self.SIDEBAR_DEFAULT_WIDTH, 800])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        layout.addWidget(self.splitter)
        return tab

    def _create_readonly_table(self, columns: Optional[List[str]] = None) -> QTableWidget:
        table = QTableWidget(0, len(columns or []))
        if columns:
            table.setHorizontalHeaderLabels(columns)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _build_capex_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.capex_status_label = QLabel("CAPEX report not loaded.")
        self.capex_status_label.setWordWrap(True)
        layout.addWidget(self.capex_status_label)

        layout.addWidget(QLabel("Summary"))
        self.capex_summary_table = self._create_readonly_table(["Field", "Value"])
        layout.addWidget(self.capex_summary_table, 1)

        layout.addWidget(QLabel("Block Summary"))
        self.capex_blocks_table = self._create_readonly_table([])
        layout.addWidget(self.capex_blocks_table, 2)

        layout.addWidget(QLabel("Entries"))
        self.capex_entries_table = self._create_readonly_table([])
        layout.addWidget(self.capex_entries_table, 3)
        return tab

    def _build_opex_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.opex_report_status_label = QLabel("OPEX report not loaded.")
        self.opex_report_status_label.setWordWrap(True)
        layout.addWidget(self.opex_report_status_label)

        layout.addWidget(QLabel("Summary"))
        self.opex_summary_table = self._create_readonly_table(["Field", "Value"])
        layout.addWidget(self.opex_summary_table, 1)

        layout.addWidget(QLabel("Items"))
        self.opex_items_table = self._create_readonly_table([])
        layout.addWidget(self.opex_items_table, 3)
        return tab

    def _build_lcoh_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.lcoh_status_label = QLabel("LCOH report not loaded.")
        self.lcoh_status_label.setWordWrap(True)
        layout.addWidget(self.lcoh_status_label)

        layout.addWidget(QLabel("Summary"))
        self.lcoh_summary_table = self._create_readonly_table(["Field", "Value"])
        layout.addWidget(self.lcoh_summary_table, 1)

        layout.addWidget(QLabel("Variants"))
        self.lcoh_variants_table = self._create_readonly_table([])
        layout.addWidget(self.lcoh_variants_table, 2)

        layout.addWidget(QLabel("Pathway Metrics"))
        self.lcoh_pathways_table = self._create_readonly_table([])
        layout.addWidget(self.lcoh_pathways_table, 2)

        layout.addWidget(QLabel("LCOH Breakdown"))
        self.lcoh_breakdown_table = self._create_readonly_table([])
        layout.addWidget(self.lcoh_breakdown_table, 2)

        layout.addWidget(QLabel("Warnings"))
        self.lcoh_warnings_table = self._create_readonly_table([])
        layout.addWidget(self.lcoh_warnings_table, 1)
        return tab

    def set_results_source(
        self,
        output_dir: Path | str | None,
        scenarios_dir: Path | str | None = None,
    ) -> None:
        """Set report-file source directories used by CAPEX/OPEX/LCOH subtabs."""
        self._results_output_dir = Path(str(output_dir)).resolve() if output_dir else None
        self._results_scenarios_dir = Path(str(scenarios_dir)).resolve() if scenarios_dir else None
        self.refresh_results_tables()

    def _stringify_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            if abs(value) >= 1000:
                return f"{value:,.4f}".rstrip("0").rstrip(".")
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value)

    def _populate_summary_table(self, table: QTableWidget, rows: List[Tuple[str, Any]]) -> None:
        table.clear()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Field", "Value"])
        table.setRowCount(len(rows))
        for row_idx, (field_name, value) in enumerate(rows):
            table.setItem(row_idx, 0, QTableWidgetItem(str(field_name)))
            table.setItem(row_idx, 1, QTableWidgetItem(self._stringify_value(value)))

    def _populate_detail_table(self, table: QTableWidget, payload: Optional[TablePayload]) -> None:
        table.clear()
        if payload is None:
            table.setColumnCount(0)
            table.setRowCount(0)
            return
        table.setColumnCount(len(payload.columns))
        table.setHorizontalHeaderLabels(payload.columns)
        table.setRowCount(len(payload.rows))
        for row_idx, row_values in enumerate(payload.rows):
            for col_idx, value in enumerate(row_values):
                table.setItem(row_idx, col_idx, QTableWidgetItem(self._stringify_value(value)))

    def _clear_results_table_set(self, *tables: QTableWidget) -> None:
        for table in tables:
            table.clear()
            table.setRowCount(0)
            table.setColumnCount(0)

    def _status_text(self, name: str, data: ReportTableData) -> str:
        source = str(data.source_path) if data.source_path else "(not found)"
        if data.status == "ok":
            return f"{name}: loaded from {source}"
        if data.status == "missing":
            return f"{name}: {data.message} (searched from {source})"
        return f"{name}: {data.message} (source: {source})"

    def refresh_results_tables(self) -> None:
        """Reload CAPEX/OPEX/LCOH table data from report files."""
        capex = load_capex_data(self._results_output_dir, self._results_scenarios_dir)
        self.capex_status_label.setText(self._status_text("CAPEX", capex))
        if capex.status == "ok":
            self._populate_summary_table(self.capex_summary_table, capex.summary_rows)
            self._populate_detail_table(self.capex_blocks_table, capex.tables.get("blocks"))
            self._populate_detail_table(self.capex_entries_table, capex.tables.get("entries"))
        else:
            self._populate_summary_table(self.capex_summary_table, [])
            self._clear_results_table_set(self.capex_blocks_table, self.capex_entries_table)

        opex = load_opex_data(self._results_output_dir, self._results_scenarios_dir)
        self.opex_report_status_label.setText(self._status_text("OPEX", opex))
        if opex.status == "ok":
            self._populate_summary_table(self.opex_summary_table, opex.summary_rows)
            self._populate_detail_table(self.opex_items_table, opex.tables.get("items"))
        else:
            self._populate_summary_table(self.opex_summary_table, [])
            self._clear_results_table_set(self.opex_items_table)

        lcoh = load_lcoh_data(self._results_output_dir, self._results_scenarios_dir)
        self.lcoh_status_label.setText(self._status_text("LCOH", lcoh))
        if lcoh.status == "ok":
            self._populate_summary_table(self.lcoh_summary_table, lcoh.summary_rows)
            self._populate_detail_table(self.lcoh_variants_table, lcoh.tables.get("variants"))
            self._populate_detail_table(self.lcoh_pathways_table, lcoh.tables.get("pathways"))
            self._populate_detail_table(self.lcoh_breakdown_table, lcoh.tables.get("breakdown"))
            self._populate_detail_table(self.lcoh_warnings_table, lcoh.tables.get("warnings"))
        else:
            self._populate_summary_table(self.lcoh_summary_table, [])
            self._clear_results_table_set(
                self.lcoh_variants_table,
                self.lcoh_pathways_table,
                self.lcoh_breakdown_table,
                self.lcoh_warnings_table,
            )

    def _theme_tokens(self) -> dict:
        """Return report-widget colors derived from ThemeManager/palette."""
        palette = QApplication.instance().palette() if QApplication.instance() else self.palette()
        is_dark = palette.color(QPalette.Window).lightness() < 128
        if is_dark:
            p = ThemeManager.DARK_PALETTE
            return {
                "frame_bg": p["surface"],
                "border": p["border"],
                "text": p["foreground"],
                "muted_text": p["comment"],
                "accent": p["blue"],
                "accent_hover": p["button_hover"],
                "button_text": "#ffffff",
                "error": p["red"],
            }

        return {
            "frame_bg": palette.color(QPalette.Base).name(),
            "border": palette.color(QPalette.Mid).name(),
            "text": palette.color(QPalette.Text).name(),
            "muted_text": palette.color(QPalette.Mid).name(),
            "accent": palette.color(QPalette.Highlight).name(),
            "accent_hover": palette.color(QPalette.LinkVisited).name(),
            "button_text": palette.color(QPalette.HighlightedText).name(),
            "error": "#b00020",
        }
    
    def _populate_tree(self):
        """Populate tree with folder hierarchy."""
        from h2_plant.visualization.graph_catalog import GRAPH_REGISTRY
        
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
                        # Use .get() method and attribute access
                        metadata = GRAPH_REGISTRY.get(graph_id)
                        child_item = QTreeWidgetItem([metadata.title])
                        child_item.setFlags(child_item.flags() | Qt.ItemIsUserCheckable)
                        child_item.setCheckState(0, folder_state)
                        child_item.setData(0, Qt.UserRole, graph_id)
                        child_item.setToolTip(0, metadata.description)
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
        theme = self._theme_tokens()
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
        self._progress_label.setStyleSheet(
            f"color: {theme['muted_text']}; font-size: 14px;"
        )
        self._progress_label.setAlignment(Qt.AlignCenter)
        self.graphs_layout.addWidget(self._progress_label)
        
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximum(len(checked_ids))
        self._progress_bar.setValue(0)
        self.graphs_layout.addWidget(self._progress_bar)
        self.graphs_layout.addStretch()
        
        # Cancel any running worker before starting a new one
        if hasattr(self, '_generation_worker') and self._generation_worker is not None and self._generation_worker.isRunning():
            self._generation_worker.request_stop()
            self._generation_worker.wait(3000)  # wait up to 3s for current graph to finish
        
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
        from h2_plant.visualization.graph_catalog import GRAPH_REGISTRY
        theme = self._theme_tokens()
        
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
            frame.setStyleSheet(
                f"""
                QFrame {{
                    background-color: {theme["frame_bg"]};
                    border: 1px solid {theme["border"]};
                    border-radius: 6px;
                    padding: 5px;
                }}
                """
            )
            frame_layout = QVBoxLayout(frame)
            
            # Graph title
            metadata = GRAPH_REGISTRY.get(graph_id)
            title_text = metadata.title if metadata else graph_id
            title = QLabel(title_text)
            title.setStyleSheet(
                f"font-weight: bold; font-size: 14px; color: {theme['text']};"
            )
            title.setAlignment(Qt.AlignCenter)
            frame_layout.addWidget(title)
            
            filepath = self.image_paths.get(graph_id)
            if not filepath:
                # Placeholder for not generated yet
                placeholder = QLabel("Not generated yet.\nClick 'Refresh Graphs' to generate.")
                placeholder.setStyleSheet(
                    f"color: {theme['muted_text']}; font-style: italic; padding: 20px;"
                )
                placeholder.setAlignment(Qt.AlignCenter)
                placeholder.setMinimumHeight(200)
                frame_layout.addWidget(placeholder)
                self.graphs_layout.addWidget(frame)
                continue
            
            # Check if this is a Plotly HTML file
            if filepath.endswith('.html'):
                # Create a clickable link to open in browser
                link_btn = QPushButton(f"🔗 Open Interactive Chart: {title_text}")
                link_btn.setStyleSheet(
                    f"""
                    QPushButton {{
                        background-color: {theme["accent"]};
                        color: {theme["button_text"]};
                        border: none;
                        border-radius: 4px;
                        padding: 12px 20px;
                        font-size: 13px;
                    }}
                    QPushButton:hover {{
                        background-color: {theme["accent_hover"]};
                    }}
                    """
                )
                link_btn.setMinimumHeight(50)
                _path = filepath  # Capture for lambda
                link_btn.clicked.connect(lambda checked, p=_path: self._open_html_graph(p))
                frame_layout.addWidget(link_btn)
                self.graphs_layout.addWidget(frame)
                continue
            
            # Image display with zoom support (PNG)
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
                image_label.setStyleSheet(f"color: {theme['error']};")
            
            image_label.setAlignment(Qt.AlignCenter)
            frame_layout.addWidget(image_label)
            
            self.image_labels[graph_id] = image_label
            self.graphs_layout.addWidget(frame)
        
        self.graphs_layout.addStretch()
    
    def _show_message(self, text, color):
        theme = self._theme_tokens()
        color_map = {
            "gray": theme["muted_text"],
            "red": theme["error"],
        }
        self.no_data_label = QLabel(text)
        self.no_data_label.setAlignment(Qt.AlignCenter)
        self.no_data_label.setStyleSheet(
            f"color: {color_map.get(color, color)}; font-size: 14px; padding: 50px;"
        )
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

    def _schedule_resize_refresh(self):
        """Debounce expensive image-rescale refreshes during window resizing."""
        if self._resize_refresh_active:
            return
        if not self.image_paths:
            return
        if not self._get_checked_graph_ids():
            return
        if self._generation_worker is not None and self._generation_worker.isRunning():
            return
        self._resize_refresh_timer.start(120)

    def _refresh_images_after_resize(self):
        """Repaint graph images at the current viewport width after resize settles."""
        if self._resize_refresh_active:
            return
        if not self.image_paths:
            return
        if self._generation_worker is not None and self._generation_worker.isRunning():
            return

        self._resize_refresh_active = True
        try:
            self._display_selected_graphs()
        finally:
            self._resize_refresh_active = False

    def resizeEvent(self, event):
        """Queue graph-rescale refreshes when the report widget size changes."""
        super().resizeEvent(event)
        self._schedule_resize_refresh()
    
    def eventFilter(self, obj, event):
        """Filter events to prevent scrolling when CTRL is held."""
        if obj == self.scroll_area.viewport():
            if event.type() == QEvent.Resize:
                self._schedule_resize_refresh()
            if event.type() == QEvent.Wheel and event.modifiers() == Qt.ControlModifier:
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
    
    def _open_html_graph(self, filepath):
        """Open an HTML (Plotly) graph in the system browser."""
        import webbrowser
        webbrowser.open(f'file://{filepath}')
    
    def cleanup(self):
        """Clean up temp directory."""
        import shutil
        if self._generation_worker is not None and self._generation_worker.isRunning():
            self._generation_worker.request_stop()
            if not self._generation_worker.wait(3000):
                logger.warning("Graph generation worker did not stop before cleanup.")
        self._resize_refresh_timer.stop()
        try:
            shutil.rmtree(self._temp_dir)
        except Exception as exc:
            logger.warning("Failed to remove temporary graph directory '%s': %s", self._temp_dir, exc)



def _infer_flow_type(port_name: str) -> FlowType:
    """Infer flow type from port name convention."""
    name = port_name.lower()
    if any(k in name for k in ('signal', 'control', 'demand')):
        return FlowType.SIGNAL
    if any(k in name for k in ('water', 'steam', 'drain', 'makeup', 'ultrapure')):
        return FlowType.WATER
    if any(k in name for k in ('o2', 'oxygen')):
        return FlowType.OXYGEN
    if any(k in name for k in ('power', 'electric', 'grid')):
        return FlowType.ELECTRICITY
    if any(k in name for k in ('h2', 'hydrogen', 'purified', 'compressed_h2')):
        return FlowType.HYDROGEN
    if any(k in name for k in ('gas', 'inlet', 'feed', 'syngas', 'tail')):
        return FlowType.GAS
    if any(k in name for k in ('heat', 'thermal', 'duty', 'cooling')):
        return FlowType.HEAT
    return FlowType.STREAM  # Safe default for thermodynamic connections


class SimulationConfigDialog(QDialog):
    """Dialog to configure simulation parameters before running."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulation Configuration")
        self.setModal(True)
        self.setMinimumWidth(420)
        
        self.selected_hours = 8760  # Default to 1 year
        self._scenarios_dir = None
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # === Duration Selection ===
        duration_group = QGroupBox("Simulation Duration")
        duration_layout = QVBoxLayout(duration_group)
        
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
        
        duration_layout.addWidget(self.radio_day)
        duration_layout.addWidget(self.radio_week)
        duration_layout.addWidget(self.radio_month)
        duration_layout.addWidget(self.radio_year)
        duration_layout.addWidget(self.radio_custom)
        
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
        duration_layout.addLayout(custom_layout)
        
        layout.addWidget(duration_group)
        
        # === Simulation Settings ===
        settings_group = QGroupBox("Simulation Settings")
        settings_layout = QFormLayout(settings_group)
        
        # Dispatch Strategy
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "REFERENCE_HYBRID",
            "SOEC_ONLY",
            "ECONOMIC_SPOT"
        ])
        self.strategy_combo.setCurrentText("REFERENCE_HYBRID")
        settings_layout.addRow("Dispatch Strategy:", self.strategy_combo)
        
        # Storage Control Mode
        self.storage_mode_combo = QComboBox()
        self.storage_mode_combo.addItems(["SCHMITT_TRIGGER", "MPC"])
        self.storage_mode_combo.setCurrentText("SCHMITT_TRIGGER")
        settings_layout.addRow("Storage Control:", self.storage_mode_combo)
        
        layout.addWidget(settings_group)
        
        # === Scenarios Directory ===
        scenario_group = QGroupBox("Scenario Configuration (Optional)")
        scenario_layout = QVBoxLayout(scenario_group)
        
        dir_layout = QHBoxLayout()
        self.scenario_dir_edit = QLineEdit()
        self.scenario_dir_edit.setPlaceholderText("Use graph nodes (no scenario dir)")
        self.scenario_dir_edit.setReadOnly(True)
        dir_layout.addWidget(self.scenario_dir_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_scenarios_dir)
        dir_layout.addWidget(browse_btn)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_scenarios_dir)
        dir_layout.addWidget(clear_btn)
        
        scenario_layout.addLayout(dir_layout)
        
        # Override checkbox
        self.override_check = QCheckBox("Override scenario defaults with dialog settings")
        self.override_check.setChecked(True)
        self.override_check.setToolTip(
            "When unchecked, duration/strategy/storage settings come from the YAML files.\n"
            "When checked, the values selected above override the scenario config."
        )
        scenario_layout.addWidget(self.override_check)
        
        layout.addWidget(scenario_group)
        
        # Connect signals
        self.radio_custom.toggled.connect(self._toggle_custom)
        self.scenario_dir_edit.textChanged.connect(self._on_scenario_dir_changed)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setText("Run Simulation")
        
        layout.addWidget(buttons)
        
    def _toggle_custom(self, checked):
        self.custom_spin.setEnabled(checked)
    
    def _browse_scenarios_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Scenarios Directory", "",
            QFileDialog.ShowDirsOnly
        )
        if dir_path:
            self._scenarios_dir = dir_path
            self.scenario_dir_edit.setText(dir_path)
    
    def _clear_scenarios_dir(self):
        self._scenarios_dir = None
        self.scenario_dir_edit.clear()
    
    def _on_scenario_dir_changed(self, text):
        """When scenario dir is set, default to not overriding."""
        has_dir = bool(text.strip())
        self.override_check.setChecked(not has_dir)
        
    def get_duration_hours(self):
        if self.radio_custom.isChecked():
            return self.custom_spin.value()
        return self.btn_group.checkedId()
    
    def get_strategy(self) -> str:
        return self.strategy_combo.currentText()
    
    def get_scenarios_dir(self):
        return self._scenarios_dir
    
    def get_storage_control_mode(self) -> str:
        return self.storage_mode_combo.currentText()
    
    def is_override_enabled(self) -> bool:
        return self.override_check.isChecked()


class PlantEditorWindow(QMainWindow):
    """H2 Plant Configuration Editor - Full Version."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("H2 Plant Configuration Editor")
        self._configure_startup_window()

        # Window mode state (for clean fullscreen round-trip)
        self._windowed_geometry = None
        self._windowed_was_maximized = False
        self._fullscreen_action = None
        self.worker = None
        self._progress_dialog = None
        self._fullscreen_reflow_timer = QTimer(self)
        self._fullscreen_reflow_timer.setSingleShot(True)
        self._fullscreen_reflow_timer.timeout.connect(self._apply_window_state_reflow)
        
        # Create node graph
        self.graph = NodeGraph()
        
        # Initialize Managers
        self.topology_engine = TopologyInferenceEngine()
        self.persistence_mgr = GraphPersistenceManager(backup_dir=Path("./backups"))
        self.validator = AdvancedValidator()

        # Scenario visual import metadata (read-only source-of-truth)
        self._scenario_manifest = None
        self._scenario_economics = {}
        self._scenario_equipment_entries = []
        self._scenario_equipment_index = {}
        self._last_selected_component_id = None
        self._workspace_root = Path(__file__).resolve().parents[1] / "layouts" / "generated"
        self._configuration_dirty = False
        self._capex_expanded = False
        self._opex_expanded = False
        self._capex_editing = False
        self._opex_editing = False
        self._capex_original_text = ""
        self._opex_original_text = ""
        self._capex_current_text = ""
        self._opex_current_text = ""
        self._active_capex_path = None
        self._active_opex_path = None
        self._capex_valid = False
        self._opex_valid = False

        # Template authoring mode state
        self._template_mode = False
        self._generated_bundle_dir = None          # Path to last-exported bundle
        self._template_source_manifest = None      # original scenarios/ reference
        self._economics_editing_connected = False   # guard for cellChanged signal

        # Zone overlay state (visual swimlane rectangles)
        self._visual_layout: dict | None = None    # visual_layout metadata from topology_analysis
        self._zone_overlay_items: list = []         # QGraphicsItem refs for cleanup
        
        # Validation Timer
        self.validation_timer = QTimer()
        self.validation_timer.timeout.connect(self.run_validation_silent)
        
        # Set central widget with Tabs
        self.central_tabs = QTabWidget()
        self.central_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.central_tabs.setMinimumSize(0, 0)
        self.graph.widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graph.widget.setMinimumSize(0, 0)
        
        # Tab 1: Graph Editor
        self.central_tabs.addTab(self.graph.widget, "Run Simulation")

        # Tab 2: Economics
        self.economics_tab = self._build_economics_tab()
        self.central_tabs.addTab(self.economics_tab, "Economics")

        # Tab 3: Configuration
        self.configuration_tab = self._build_configuration_tab()
        self.central_tabs.addTab(self.configuration_tab, "Configuration")
        
        # Tab 4: Simulation Report
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
        self._enforce_resizable_window_constraints()

        # Keep equipment panel synced to selected node.
        # Primary: signal-based (instant, zero-cost when idle)
        self.graph.node_selection_changed.connect(
            lambda *_: self._refresh_equipment_panel_selection()
        )
        # Fallback: slow poll for edge cases (deselect-via-background, etc.)
        self._selection_poll_timer = QTimer(self)
        self._selection_poll_timer.timeout.connect(self._refresh_equipment_panel_selection)
        self._selection_poll_timer.start(2000)  # 2s fallback, was 300ms

        self._refresh_workspace_tabs()
        
        # Apply theme
        ThemeManager.apply_theme(self, QApplication.instance(), "dark")
        self.show()

    def _configure_startup_window(self):
        """Size and center the startup window relative to available screen geometry."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(1400, 900)
            return

        available = screen.availableGeometry()
        target_width = min(available.width(), max(960, int(available.width() * 0.9)))
        target_height = min(available.height(), max(680, int(available.height() * 0.9)))

        self.resize(target_width, target_height)
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    def _enforce_resizable_window_constraints(self):
        """Ensure the main window remains fully resizable in normal mode.

        Note: ``Qt.MSWindowsFixedSizeDialogHint`` is intentionally **not**
        touched here.  Calling ``setWindowFlag`` triggers native-window
        recreation on X11/Wayland, which resets the WM size hints and can
        permanently lock the window to its current size.  The hint is
        Windows-only anyway, so skipping it is a no-op on Debian.
        """
        self.setMaximumSize(16777215, 16777215)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def register_all_nodes(self):
        """Register all typed node classes (1:1 backend coverage) plus fallback."""
        self.node_classes = [
            # Electrolysis / Production
            PEMStackNode,
            SOECStackNode,
            RectifierNode,
            # Thermal
            ChillerNode,
            DryCoolerNode,
            InterchangerNode,
            ElectricBoilerNode,
            AttemperatorNode,
            CoolingManagerNode,
            # Separation
            CoalescerNode,
            KnockOutDrumNode,
            PSAUnitNode,
            DeoxoReactorNode,
            HydrogenMultiCycloneNode,
            SeparationTankNode,
            SyngasPSANode,
            # Mixing / Flow
            MixerNode,
            ValveNode,
            StreamSplitterNode,
            DrainRecorderMixerNode,
            SignalMakeupMixerNode,
            ProportionalMakeupMixerNode,
            OxygenMakeupNode,
            # Water
            WaterPurifierNode,
            UltraPureWaterTankNode,
            ExternalWaterSourceNode,
            WaterPumpThermodynamicNode,
            # Storage / Delivery
            DetailedTankNode,
            DischargeStationNode,
            CompressorSingleNode,
            # Reforming
            IntegratedATRPlantNode,
            ATRBoilerNode,
            BiogasSourceNode,
            # Fallback
            ScenarioComponentNode,
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

        # Scenario Economics (read-only)
        self.scenario_economics_table = QTableWidget(0, 2)
        self.scenario_economics_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.scenario_economics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.scenario_economics_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.scenario_economics_table.verticalHeader().setVisible(False)
        self.scenario_economics_table.horizontalHeader().setStretchLastSection(True)

        self.scenario_economics_dock = QDockWidget("Scenario Economics", self)
        self.scenario_economics_dock.setWidget(self.scenario_economics_table)
        self.addDockWidget(Qt.RightDockWidgetArea, self.scenario_economics_dock)
        self.scenario_economics_dock.hide()

        # Equipment Mapping (read-only, linked to selected node)
        self.equipment_mapping_table = QTableWidget(0, 7)
        self.equipment_mapping_table.setHorizontalHeaderLabels([
            "Tag", "Block", "Name", "Capacity", "Unit", "Cost Source", "Topology IDs"
        ])
        self.equipment_mapping_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.equipment_mapping_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.equipment_mapping_table.verticalHeader().setVisible(False)
        self.equipment_mapping_table.horizontalHeader().setStretchLastSection(True)

        self.equipment_mapping_dock = QDockWidget("Equipment Mapping", self)
        self.equipment_mapping_dock.setWidget(self.equipment_mapping_table)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.equipment_mapping_dock)
        self.equipment_mapping_dock.hide()

        # Imported backend params not mapped to typed GUI fields (read-only)
        self.imported_params_table = QTableWidget(0, 2)
        self.imported_params_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.imported_params_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.imported_params_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.imported_params_table.verticalHeader().setVisible(False)
        self.imported_params_table.horizontalHeader().setStretchLastSection(True)

        self.imported_params_dock = QDockWidget("Imported Params", self)
        self.imported_params_dock.setWidget(self.imported_params_table)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.imported_params_dock)
        self.imported_params_dock.hide()
    
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

        open_prebuilt_menu = file_menu.addMenu("Open Prebuilt Visual Twin")
        for variant_label, variant_cfg in PREBUILT_VISUAL_VARIANTS.items():
            open_prebuilt_action = QAction(variant_label, self)
            open_prebuilt_action.triggered.connect(
                partial(
                    self.open_prebuilt_visual_twin,
                    variant_label=variant_label,
                    **variant_cfg,
                )
            )
            open_prebuilt_menu.addAction(open_prebuilt_action)

        import_scenario_action = QAction("Import Scenario Visual...", self)
        import_scenario_action.triggered.connect(self.import_scenario_visual)
        file_menu.addAction(import_scenario_action)
        
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

        view_menu.addSeparator()

        auto_layout_action = QAction("Auto-Layout (Industrial PFD)", self)
        auto_layout_action.setStatusTip("Recalculate industrial plant layout and redraw grouped overlays")
        auto_layout_action.triggered.connect(self._run_auto_layout_report_mode)
        view_menu.addAction(auto_layout_action)

        view_menu.addSeparator()

        fullscreen_action = QAction("Full Screen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.setCheckable(True)
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        self._fullscreen_action = fullscreen_action
        self._sync_fullscreen_action_state()

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

        # Explicit top-right window controls (additive to native title bar controls).
        window_controls = QWidget(menubar)
        controls_layout = QHBoxLayout(window_controls)
        controls_layout.setContentsMargins(4, 2, 4, 2)
        controls_layout.setSpacing(4)

        self.minimize_button = QToolButton(window_controls)
        self.minimize_button.setText("—")
        self.minimize_button.setToolTip("Minimize")
        self.minimize_button.setFixedSize(28, 22)
        self.minimize_button.clicked.connect(self.showMinimized)
        controls_layout.addWidget(self.minimize_button)

        self.window_mode_button = QToolButton(window_controls)
        self.window_mode_button.setFixedSize(28, 22)
        self.window_mode_button.clicked.connect(
            lambda _checked=False: self._toggle_fullscreen()
        )
        controls_layout.addWidget(self.window_mode_button)
        self._sync_window_mode_button_state()

        window_controls.setStyleSheet(
            """
            QToolButton {
                border: 1px solid #3e4451;
                border-radius: 3px;
                background-color: #3b4048;
                color: #abb2bf;
                font-weight: bold;
                padding: 0px;
            }
            QToolButton:hover {
                background-color: #4b5263;
            }
            """
        )
        menubar.setCornerWidget(window_controls, Qt.TopRightCorner)



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
            logger.warning(f"Could not setup context menu: {e}")
    
    def setup_keyboard_shortcuts(self):
        """Setup additional keyboard shortcuts."""
        QShortcut(QKeySequence(Qt.Key_F11), self, lambda: self._toggle_fullscreen())
        QShortcut(QKeySequence(Qt.Key_Escape), self, self._handle_escape_pressed)

    def _handle_escape_pressed(self):
        """Exit fullscreen quickly without affecting normal window mode."""
        if self.isFullScreen():
            self._exit_fullscreen()

    def _enter_fullscreen(self):
        """Enter fullscreen and preserve the current windowed state."""
        if self.isFullScreen():
            return

        self._windowed_was_maximized = self.isMaximized()
        if not self._windowed_was_maximized:
            self._windowed_geometry = self.geometry()

        # Save dock visibility so we can restore later, then hide them
        # so the central canvas can occupy the full screen width.
        self._saved_dock_visibility = {}
        for dock in self.findChildren(QDockWidget):
            self._saved_dock_visibility[dock] = dock.isVisible()
            dock.hide()

        self.showFullScreen()
        self._sync_fullscreen_action_state()
        self._schedule_window_state_reflow(80)

    def _exit_fullscreen(self):
        """Exit fullscreen and restore the previous windowed/maximized state."""
        if not self.isFullScreen():
            self._sync_fullscreen_action_state()
            return

        self.showNormal()
        if self._windowed_was_maximized:
            self.showMaximized()
        elif self._windowed_geometry is not None and self._windowed_geometry.isValid():
            self.setGeometry(self._windowed_geometry)

        # Restore dock visibility saved during _enter_fullscreen
        for dock, was_visible in getattr(self, '_saved_dock_visibility', {}).items():
            dock.setVisible(was_visible)
        self._saved_dock_visibility = {}

        self._enforce_resizable_window_constraints()
        self._sync_fullscreen_action_state()
        self._schedule_window_state_reflow()

    def _toggle_fullscreen(self, checked=None):
        """Toggle fullscreen from action state or shortcut."""
        want_fullscreen = (not self.isFullScreen()) if checked is None else bool(checked)
        if want_fullscreen:
            self._enter_fullscreen()
        else:
            self._exit_fullscreen()

    def _sync_fullscreen_action_state(self):
        """Keep the View->Full Screen action synchronized with actual window state."""
        if self._fullscreen_action is None:
            return
        is_fullscreen = self.isFullScreen()
        if self._fullscreen_action.isChecked() != is_fullscreen:
            self._fullscreen_action.blockSignals(True)
            self._fullscreen_action.setChecked(is_fullscreen)
            self._fullscreen_action.blockSignals(False)
        self._sync_window_mode_button_state()

    def _sync_window_mode_button_state(self):
        """Keep the top-right fullscreen/window button icon and tooltip in sync."""
        if not hasattr(self, "window_mode_button") or self.window_mode_button is None:
            return
        if self.isFullScreen():
            icon = self.style().standardIcon(QStyle.SP_TitleBarNormalButton)
            tooltip = "Exit Full Screen"
        else:
            icon = self.style().standardIcon(QStyle.SP_TitleBarMaxButton)
            tooltip = "Enter Full Screen"
        self.window_mode_button.setIcon(icon)
        self.window_mode_button.setToolTip(tooltip)

    def _schedule_window_state_reflow(self, delay_ms=0):
        """Debounce window/canvas reflow after fullscreen and state transitions."""
        self._fullscreen_reflow_timer.start(max(0, int(delay_ms)))

    def _apply_window_state_reflow(self):
        """Force central layout and graph view to recalculate geometry.

        Performs two passes: an immediate layout activation followed by a
        secondary 150 ms delayed pass.  This catches late WM geometry
        propagation that typical Linux X11/Wayland compositors may defer.
        """
        self._apply_reflow_pass()

        # Secondary delayed pass — catches geometry changes the WM pushes
        # after the initial showFullScreen / showNormal completes.
        QTimer.singleShot(150, self._apply_reflow_pass)

    def _apply_reflow_pass(self):
        """Single geometry-reflow pass for all managed widgets."""
        # In fullscreen, explicitly resize the window to the available
        # screen geometry so the WM has no ambiguity.
        if self.isFullScreen():
            screen = self.screen() or QApplication.primaryScreen()
            if screen is not None:
                geo = screen.geometry()
                if self.geometry() != geo:
                    self.setGeometry(geo)

        central_widget = self.centralWidget()
        if central_widget is not None:
            central_widget.updateGeometry()
            if central_widget.layout() is not None:
                central_widget.layout().activate()

        if hasattr(self, "central_tabs") and self.central_tabs is not None:
            self.central_tabs.updateGeometry()
            self.central_tabs.update()

        graph_widget = getattr(self.graph, "widget", None)
        if graph_widget is not None:
            graph_widget.updateGeometry()
            graph_widget.update()

        viewer = None
        try:
            viewer = self.graph.viewer()
        except Exception as exc:
            logger.debug("Unable to access graph viewer for reflow: %s", exc)

        if viewer is not None:
            viewer.updateGeometry()
            viewport = viewer.viewport() if hasattr(viewer, "viewport") else None
            if viewport is not None:
                viewport.updateGeometry()
                viewport.update()
            scene = viewer.scene() if hasattr(viewer, "scene") else None
            if scene is not None:
                scene.update()
            viewer.update()

        if hasattr(self, "report_widget") and self.report_widget is not None:
            self.report_widget.updateGeometry()
            if hasattr(self.report_widget, "_schedule_resize_refresh"):
                self.report_widget._schedule_resize_refresh()

        self._sync_fullscreen_action_state()

    def changeEvent(self, event):
        """Synchronize fullscreen action when window state changes externally."""
        if event.type() == QEvent.WindowStateChange:
            self._sync_fullscreen_action_state()
            self._schedule_window_state_reflow()
        super().changeEvent(event)

    def closeEvent(self, event):
        """Stop background work and release temp assets on application close."""
        self.validation_timer.stop()
        if hasattr(self, "_selection_poll_timer"):
            self._selection_poll_timer.stop()

        self._stop_active_simulation_worker(wait_ms=5000)
        self._fullscreen_reflow_timer.stop()

        if self._progress_dialog is not None:
            self._progress_dialog.hide()
            self._progress_dialog.deleteLater()
            self._progress_dialog = None

        if hasattr(self, "report_widget") and self.report_widget is not None:
            self.report_widget.cleanup()

        super().closeEvent(event)

    def _get_active_workspace_dir(self):
        if not self._scenario_manifest:
            return None
        scenarios_dir = self._scenario_manifest.get("scenarios_dir")
        if not scenarios_dir:
            return None
        return Path(str(scenarios_dir))

    def _is_copy_manifest(self):
        manifest = self._scenario_manifest or {}
        if manifest.get("workspace_generated_at"):
            return True
        workspace_dir = self._get_active_workspace_dir()
        if not workspace_dir:
            return False
        try:
            workspace_dir.resolve().relative_to(self._workspace_root.resolve())
            return True
        except ValueError:
            return False

    def _build_economics_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.econ_workspace_label = QLabel("Active workspace: (none)")
        layout.addWidget(self.econ_workspace_label)

        files_group = QGroupBox("Economics Files")
        files_layout = QFormLayout(files_group)

        self.opex_status_label = QLabel("OPEX: not loaded")
        opex_row = QHBoxLayout()
        opex_select_btn = QPushButton("Select OPEX File...")
        opex_select_btn.clicked.connect(self._select_opex_file_for_workspace)
        opex_row.addWidget(self.opex_status_label, 1)
        opex_row.addWidget(opex_select_btn)
        files_layout.addRow("OPEX (`opex_config.yaml`):", opex_row)

        self.equipment_status_label = QLabel("Equipment mappings: not loaded")
        equipment_row = QHBoxLayout()
        equipment_select_btn = QPushButton("Select Equipment File...")
        equipment_select_btn.clicked.connect(self._select_equipment_file_for_workspace)
        equipment_row.addWidget(self.equipment_status_label, 1)
        equipment_row.addWidget(equipment_select_btn)
        files_layout.addRow("Equipment (`equipment_mappings.yaml`):", equipment_row)
        layout.addWidget(files_group)

        capex_group = QGroupBox("CAPEX (`equipment_mappings.yaml`)")
        capex_layout = QVBoxLayout(capex_group)
        capex_header = QHBoxLayout()
        self.capex_path_label = QLabel("(missing)")
        self.capex_path_label.setWordWrap(True)
        capex_header.addWidget(self.capex_path_label, 1)
        self.capex_expand_btn = QPushButton("Expand")
        self.capex_expand_btn.clicked.connect(self._toggle_capex_expand)
        capex_header.addWidget(self.capex_expand_btn)
        self.capex_edit_btn = QPushButton("Edit")
        self.capex_edit_btn.clicked.connect(self._enter_capex_edit_mode)
        capex_header.addWidget(self.capex_edit_btn)
        self.capex_cancel_btn = QPushButton("Cancel")
        self.capex_cancel_btn.clicked.connect(self._cancel_capex_edit)
        self.capex_cancel_btn.setVisible(False)
        capex_header.addWidget(self.capex_cancel_btn)
        self.capex_save_btn = QPushButton("Save changes")
        self.capex_save_btn.clicked.connect(self._save_capex_edit)
        self.capex_save_btn.setVisible(False)
        capex_header.addWidget(self.capex_save_btn)
        capex_layout.addLayout(capex_header)
        self.capex_editor = QPlainTextEdit()
        self.capex_editor.setReadOnly(True)
        self.capex_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.capex_editor.textChanged.connect(self._on_capex_text_changed)
        capex_layout.addWidget(self.capex_editor)
        layout.addWidget(capex_group, 1)

        opex_group = QGroupBox("OPEX (`opex_config.yaml`)")
        opex_layout = QVBoxLayout(opex_group)
        opex_header = QHBoxLayout()
        self.opex_path_label = QLabel("(missing)")
        self.opex_path_label.setWordWrap(True)
        opex_header.addWidget(self.opex_path_label, 1)
        self.opex_expand_btn = QPushButton("Expand")
        self.opex_expand_btn.clicked.connect(self._toggle_opex_expand)
        opex_header.addWidget(self.opex_expand_btn)
        self.opex_edit_btn = QPushButton("Edit")
        self.opex_edit_btn.clicked.connect(self._enter_opex_edit_mode)
        opex_header.addWidget(self.opex_edit_btn)
        self.opex_cancel_btn = QPushButton("Cancel")
        self.opex_cancel_btn.clicked.connect(self._cancel_opex_edit)
        self.opex_cancel_btn.setVisible(False)
        opex_header.addWidget(self.opex_cancel_btn)
        self.opex_save_btn = QPushButton("Save changes")
        self.opex_save_btn.clicked.connect(self._save_opex_edit)
        self.opex_save_btn.setVisible(False)
        opex_header.addWidget(self.opex_save_btn)
        opex_layout.addLayout(opex_header)
        self.opex_editor = QPlainTextEdit()
        self.opex_editor.setReadOnly(True)
        self.opex_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.opex_editor.textChanged.connect(self._on_opex_text_changed)
        opex_layout.addWidget(self.opex_editor)
        layout.addWidget(opex_group, 1)

        general_group = QGroupBox("General Economic Information")
        general_form = QFormLayout(general_group)
        self.cepci_base_year_spin = QSpinBox()
        self.cepci_base_year_spin.setRange(1900, 2500)
        general_form.addRow("cepci.base_year:", self.cepci_base_year_spin)
        self.cepci_base_index_spin = QDoubleSpinBox()
        self.cepci_base_index_spin.setDecimals(6)
        self.cepci_base_index_spin.setRange(0.0, 1_000_000.0)
        general_form.addRow("cepci.base_index:", self.cepci_base_index_spin)
        self.cepci_current_year_spin = QSpinBox()
        self.cepci_current_year_spin.setRange(1900, 2500)
        general_form.addRow("cepci.current_year:", self.cepci_current_year_spin)
        self.cepci_current_index_spin = QDoubleSpinBox()
        self.cepci_current_index_spin.setDecimals(6)
        self.cepci_current_index_spin.setRange(0.0, 1_000_000.0)
        general_form.addRow("cepci.current_index:", self.cepci_current_index_spin)
        self.capacity_mode_combo = QComboBox()
        self.capacity_mode_combo.addItems(["design", "history"])
        general_form.addRow("capacity_mode:", self.capacity_mode_combo)
        layout.addWidget(general_group)

        self._set_general_econ_fields_enabled(False)
        self._set_capex_expanded(False)
        self._set_opex_expanded(False)
        self._update_capex_editor_actions()
        self._update_opex_editor_actions()
        
        layout.addStretch(1)  # Fix vertical height lock by allowing empty space to expand
        
        return tab

    def _collapsed_editor_height(self, editor, lines: int = 3) -> int:
        line_height = editor.fontMetrics().lineSpacing()
        margins = editor.contentsMargins()
        frame = int(editor.frameWidth() * 2)
        doc_margin = int(editor.document().documentMargin() * 2)
        return int(line_height * lines + margins.top() + margins.bottom() + frame + doc_margin + 6)

    def _apply_editor_expand_state(self, editor, expanded: bool):
        if expanded:
            editor.setMinimumHeight(0)
            editor.setMaximumHeight(16777215)
            editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        else:
            height = self._collapsed_editor_height(editor, lines=3)
            editor.setMinimumHeight(height)
            editor.setMaximumHeight(height)
            editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def _set_capex_expanded(self, expanded: bool):
        self._capex_expanded = bool(expanded)
        self.capex_expand_btn.setText("Collapse" if self._capex_expanded else "Expand")
        self._apply_editor_expand_state(self.capex_editor, self._capex_expanded)

    def _set_opex_expanded(self, expanded: bool):
        self._opex_expanded = bool(expanded)
        self.opex_expand_btn.setText("Collapse" if self._opex_expanded else "Expand")
        self._apply_editor_expand_state(self.opex_editor, self._opex_expanded)

    def _toggle_capex_expand(self):
        self._set_capex_expanded(not self._capex_expanded)

    def _toggle_opex_expand(self):
        self._set_opex_expanded(not self._opex_expanded)

    def _set_general_econ_fields_enabled(self, enabled: bool):
        for widget in (
            self.cepci_base_year_spin,
            self.cepci_base_index_spin,
            self.cepci_current_year_spin,
            self.cepci_current_index_spin,
            self.capacity_mode_combo,
        ):
            widget.setEnabled(enabled)

    def _set_general_econ_fields(self, info):
        self.cepci_base_year_spin.blockSignals(True)
        self.cepci_base_index_spin.blockSignals(True)
        self.cepci_current_year_spin.blockSignals(True)
        self.cepci_current_index_spin.blockSignals(True)
        self.capacity_mode_combo.blockSignals(True)
        try:
            self.cepci_base_year_spin.setValue(int(info.get("base_year", 2001)))
            self.cepci_base_index_spin.setValue(float(info.get("base_index", 397.0)))
            self.cepci_current_year_spin.setValue(int(info.get("current_year", 2025)))
            self.cepci_current_index_spin.setValue(float(info.get("current_index", 797.0)))
            mode_value = str(info.get("capacity_mode", "history")).strip().lower()
            if self.capacity_mode_combo.findText(mode_value) < 0:
                mode_value = "history"
            self.capacity_mode_combo.setCurrentText(mode_value)
        finally:
            self.cepci_base_year_spin.blockSignals(False)
            self.cepci_base_index_spin.blockSignals(False)
            self.cepci_current_year_spin.blockSignals(False)
            self.cepci_current_index_spin.blockSignals(False)
            self.capacity_mode_combo.blockSignals(False)

    def _collect_general_econ_fields(self):
        return {
            "base_year": int(self.cepci_base_year_spin.value()),
            "base_index": float(self.cepci_base_index_spin.value()),
            "current_year": int(self.cepci_current_year_spin.value()),
            "current_index": float(self.cepci_current_index_spin.value()),
            "capacity_mode": str(self.capacity_mode_combo.currentText()).strip().lower(),
        }

    def _update_capex_editor_actions(self):
        can_edit = bool(
            self._active_capex_path
            and self._active_capex_path.exists()
            and self._is_copy_manifest()
            and self._capex_valid
        )
        self.capex_edit_btn.setVisible(not self._capex_editing)
        self.capex_edit_btn.setEnabled(can_edit and not self._capex_editing)
        self.capex_cancel_btn.setVisible(self._capex_editing)
        self.capex_cancel_btn.setEnabled(self._capex_editing)
        self.capex_save_btn.setVisible(self._capex_editing)
        self.capex_save_btn.setEnabled(self._capex_editing and can_edit)
        self.capex_editor.setReadOnly(not self._capex_editing)
        self._set_general_econ_fields_enabled(self._capex_editing)

    def _update_opex_editor_actions(self):
        can_edit = bool(
            self._active_opex_path
            and self._active_opex_path.exists()
            and self._is_copy_manifest()
            and self._opex_valid
        )
        self.opex_edit_btn.setVisible(not self._opex_editing)
        self.opex_edit_btn.setEnabled(can_edit and not self._opex_editing)
        self.opex_cancel_btn.setVisible(self._opex_editing)
        self.opex_cancel_btn.setEnabled(self._opex_editing)
        self.opex_save_btn.setVisible(self._opex_editing)
        self.opex_save_btn.setEnabled(self._opex_editing and can_edit)
        self.opex_editor.setReadOnly(not self._opex_editing)

    def _on_capex_text_changed(self):
        self._capex_current_text = self.capex_editor.toPlainText()

    def _on_opex_text_changed(self):
        self._opex_current_text = self.opex_editor.toPlainText()

    def _enter_capex_edit_mode(self):
        if not self._active_capex_path or not self._active_capex_path.exists():
            QMessageBox.warning(self, "Economics", "No CAPEX file loaded in workspace.")
            return
        if not self._is_copy_manifest():
            QMessageBox.warning(
                self,
                "Economics",
                "Active scenario is not staged as a copy workspace; edit is disabled.",
            )
            return
        self._capex_original_text = self.capex_editor.toPlainText()
        self._capex_current_text = self._capex_original_text
        self._capex_editing = True
        self._update_capex_editor_actions()
        self.capex_editor.setFocus()

    def _enter_opex_edit_mode(self):
        if not self._active_opex_path or not self._active_opex_path.exists():
            QMessageBox.warning(self, "Economics", "No OPEX file loaded in workspace.")
            return
        if not self._is_copy_manifest():
            QMessageBox.warning(
                self,
                "Economics",
                "Active scenario is not staged as a copy workspace; edit is disabled.",
            )
            return
        self._opex_original_text = self.opex_editor.toPlainText()
        self._opex_current_text = self._opex_original_text
        self._opex_editing = True
        self._update_opex_editor_actions()
        self.opex_editor.setFocus()

    def _cancel_capex_edit(self):
        self._capex_editing = False
        self.capex_editor.setPlainText(self._capex_original_text)
        self._capex_current_text = self._capex_original_text
        try:
            capex_data = validate_capex_yaml_text(self._capex_original_text)
            self._set_general_econ_fields(extract_general_econ_info(capex_data))
        except Exception:
            self._set_general_econ_fields(extract_general_econ_info({}))
        self._update_capex_editor_actions()

    def _cancel_opex_edit(self):
        self._opex_editing = False
        self.opex_editor.setPlainText(self._opex_original_text)
        self._opex_current_text = self._opex_original_text
        self._update_opex_editor_actions()

    def _save_capex_edit(self):
        if not self._active_capex_path:
            QMessageBox.warning(self, "Economics", "No CAPEX file target available.")
            return
        if not self._is_copy_manifest():
            QMessageBox.warning(
                self,
                "Economics",
                "Active scenario is not staged as a copy workspace; save is disabled.",
            )
            return

        raw_text = self.capex_editor.toPlainText()
        try:
            capex_data = validate_capex_yaml_text(raw_text)
            merged = apply_general_econ_info(capex_data, self._collect_general_econ_fields())
            normalized = validate_capex_yaml_text(
                yaml.safe_dump(merged, default_flow_style=False, sort_keys=False, allow_unicode=True)
            )
        except Exception as exc:
            QMessageBox.critical(self, "Invalid CAPEX YAML", f"CAPEX validation failed:\n{exc}")
            return

        try:
            self._active_capex_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._active_capex_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    normalized,
                    handle,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
        except Exception as exc:
            QMessageBox.critical(self, "Economics", f"Failed to save CAPEX copy:\n{exc}")
            return

        self._scenario_manifest = refresh_manifest_file_hashes(dict(self._scenario_manifest or {}))
        self._capex_editing = False
        self._refresh_economics_tab()
        QMessageBox.information(self, "Economics", "CAPEX copy saved.")

    def _save_opex_edit(self):
        if not self._active_opex_path:
            QMessageBox.warning(self, "Economics", "No OPEX file target available.")
            return
        if not self._is_copy_manifest():
            QMessageBox.warning(
                self,
                "Economics",
                "Active scenario is not staged as a copy workspace; save is disabled.",
            )
            return

        raw_text = self.opex_editor.toPlainText()
        try:
            opex_data = validate_opex_yaml_text(raw_text)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid OPEX YAML", f"OPEX validation failed:\n{exc}")
            return

        try:
            self._active_opex_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._active_opex_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    opex_data,
                    handle,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
        except Exception as exc:
            QMessageBox.critical(self, "Economics", f"Failed to save OPEX copy:\n{exc}")
            return

        self._scenario_manifest = refresh_manifest_file_hashes(dict(self._scenario_manifest or {}))
        self._opex_editing = False
        self._refresh_economics_tab()
        QMessageBox.information(self, "Economics", "OPEX copy saved.")

    def _validate_opex_text(self, text):
        return validate_opex_yaml_text(text)

    def _validate_capex_text(self, text):
        return validate_capex_yaml_text(text)

    def _build_configuration_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.config_workspace_label = QLabel("Active workspace: (none)")
        layout.addWidget(self.config_workspace_label)

        form_group = QGroupBox("Simulation Configuration")
        form_layout = QFormLayout(form_group)

        self.config_timestep_spin = QDoubleSpinBox()
        self.config_timestep_spin.setDecimals(6)
        self.config_timestep_spin.setRange(0.000001, 1000.0)
        self.config_timestep_spin.valueChanged.connect(self._mark_configuration_dirty)
        form_layout.addRow("timestep_hours:", self.config_timestep_spin)

        self.config_duration_spin = QSpinBox()
        self.config_duration_spin.setRange(1, 1_000_000)
        self.config_duration_spin.valueChanged.connect(self._mark_configuration_dirty)
        form_layout.addRow("duration_hours:", self.config_duration_spin)

        self.config_start_spin = QSpinBox()
        self.config_start_spin.setRange(0, 1_000_000)
        self.config_start_spin.valueChanged.connect(self._mark_configuration_dirty)
        form_layout.addRow("start_hour:", self.config_start_spin)

        self.config_checkpoint_spin = QSpinBox()
        self.config_checkpoint_spin.setRange(0, 1_000_000)
        self.config_checkpoint_spin.valueChanged.connect(self._mark_configuration_dirty)
        form_layout.addRow("checkpoint_interval_hours:", self.config_checkpoint_spin)

        self.config_energy_edit = QLineEdit()
        self.config_energy_edit.textChanged.connect(self._mark_configuration_dirty)
        energy_row = QHBoxLayout()
        energy_row.addWidget(self.config_energy_edit, 1)
        browse_energy_btn = QPushButton("Browse...")
        browse_energy_btn.clicked.connect(lambda: self._browse_config_data_file("energy"))
        energy_row.addWidget(browse_energy_btn)
        form_layout.addRow("energy_price_file:", energy_row)

        self.config_wind_edit = QLineEdit()
        self.config_wind_edit.textChanged.connect(self._mark_configuration_dirty)
        wind_row = QHBoxLayout()
        wind_row.addWidget(self.config_wind_edit, 1)
        browse_wind_btn = QPushButton("Browse...")
        browse_wind_btn.clicked.connect(lambda: self._browse_config_data_file("wind"))
        wind_row.addWidget(browse_wind_btn)
        form_layout.addRow("wind_data_file:", wind_row)

        self.config_dispatch_combo = QComboBox()
        self.config_dispatch_combo.addItems(["REFERENCE_HYBRID", "SOEC_ONLY", "ECONOMIC_SPOT"])
        self.config_dispatch_combo.currentTextChanged.connect(self._mark_configuration_dirty)
        form_layout.addRow("dispatch_strategy:", self.config_dispatch_combo)

        self.config_storage_combo = QComboBox()
        self.config_storage_combo.addItems(["SCHMITT_TRIGGER", "MPC"])
        self.config_storage_combo.currentTextChanged.connect(self._mark_configuration_dirty)
        form_layout.addRow("storage_control_mode:", self.config_storage_combo)

        layout.addWidget(form_group)

        actions = QHBoxLayout()
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._load_configuration_from_workspace)
        actions.addWidget(reload_btn)

        save_btn = QPushButton("Save Configuration")
        save_btn.clicked.connect(self._save_configuration_to_workspace)
        actions.addWidget(save_btn)

        actions.addStretch(1)
        self.config_dirty_label = QLabel("Saved")
        actions.addWidget(self.config_dirty_label)
        layout.addLayout(actions)
        layout.addStretch(1)
        return tab

    def _mark_configuration_dirty(self, *_args):
        self._configuration_dirty = True
        if hasattr(self, "config_dirty_label"):
            self.config_dirty_label.setText("Unsaved changes")

    def _clear_configuration_dirty(self):
        self._configuration_dirty = False
        if hasattr(self, "config_dirty_label"):
            self.config_dirty_label.setText("Saved")

    def _set_form_enabled(self, enabled: bool):
        for widget in (
            self.config_timestep_spin,
            self.config_duration_spin,
            self.config_start_spin,
            self.config_checkpoint_spin,
            self.config_energy_edit,
            self.config_wind_edit,
            self.config_dispatch_combo,
            self.config_storage_combo,
        ):
            widget.setEnabled(enabled)

    def _default_simulation_config(self):
        defaults = {}
        for key, field in SimulationConfig.model_fields.items():
            if field.default is not None:
                defaults[key] = field.default
        defaults.setdefault("timestep_hours", 0.0167)
        defaults.setdefault("duration_hours", 24)
        defaults.setdefault("start_hour", 0)
        defaults.setdefault("checkpoint_interval_hours", 120)
        defaults.setdefault("energy_price_file", "../h2_plant/data/NL_Prices_2024_15min.csv")
        defaults.setdefault("wind_data_file", "../h2_plant/data/producao_horaria_turbina.csv")
        defaults.setdefault("dispatch_strategy", "ECONOMIC_SPOT")
        defaults.setdefault("storage_control_mode", "SCHMITT_TRIGGER")
        return defaults

    def _simulation_form_to_dict(self):
        return {
            "timestep_hours": float(self.config_timestep_spin.value()),
            "duration_hours": int(self.config_duration_spin.value()),
            "start_hour": int(self.config_start_spin.value()),
            "checkpoint_interval_hours": int(self.config_checkpoint_spin.value()),
            "energy_price_file": self.config_energy_edit.text().strip(),
            "wind_data_file": self.config_wind_edit.text().strip(),
            "dispatch_strategy": self.config_dispatch_combo.currentText().strip(),
            "storage_control_mode": self.config_storage_combo.currentText().strip(),
        }

    def _load_configuration_from_workspace(self):
        workspace_dir = self._get_active_workspace_dir()
        if not workspace_dir:
            self.config_workspace_label.setText("Active workspace: (none)")
            self._set_form_enabled(False)
            self._clear_configuration_dirty()
            return

        self._set_form_enabled(True)
        self.config_workspace_label.setText(f"Active workspace: {workspace_dir}")

        config_path = resolve_manifest_file(
            self._scenario_manifest,
            "simulation_config_file",
            DEFAULT_SIMULATION_FILE,
        )
        if not config_path:
            config_path = workspace_dir / DEFAULT_SIMULATION_FILE
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if not config_path.exists():
            defaults = self._default_simulation_config()
            if self._is_copy_manifest():
                with open(config_path, "w", encoding="utf-8") as handle:
                    yaml.safe_dump(defaults, handle, default_flow_style=False, sort_keys=False, allow_unicode=True)
            else:
                sim_data = defaults
                config_path = None

        if config_path:
            try:
                sim_data = load_yaml_preview(config_path)
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Configuration",
                    f"Failed to load simulation config for preview:\n{exc}",
                )
                sim_data = self._default_simulation_config()

        self.config_timestep_spin.blockSignals(True)
        self.config_duration_spin.blockSignals(True)
        self.config_start_spin.blockSignals(True)
        self.config_checkpoint_spin.blockSignals(True)
        self.config_energy_edit.blockSignals(True)
        self.config_wind_edit.blockSignals(True)
        self.config_dispatch_combo.blockSignals(True)
        self.config_storage_combo.blockSignals(True)
        try:
            defaults = self._default_simulation_config()
            merged = {**defaults, **sim_data}
            self.config_timestep_spin.setValue(float(merged["timestep_hours"]))
            self.config_duration_spin.setValue(int(merged["duration_hours"]))
            self.config_start_spin.setValue(int(merged["start_hour"]))
            self.config_checkpoint_spin.setValue(int(merged["checkpoint_interval_hours"]))
            self.config_energy_edit.setText(str(merged["energy_price_file"]))
            self.config_wind_edit.setText(str(merged["wind_data_file"]))
            dispatch_value = str(merged["dispatch_strategy"])
            if self.config_dispatch_combo.findText(dispatch_value) < 0:
                self.config_dispatch_combo.addItem(dispatch_value)
            self.config_dispatch_combo.setCurrentText(dispatch_value)
            storage_value = str(merged["storage_control_mode"])
            if self.config_storage_combo.findText(storage_value) < 0:
                self.config_storage_combo.addItem(storage_value)
            self.config_storage_combo.setCurrentText(storage_value)
        finally:
            self.config_timestep_spin.blockSignals(False)
            self.config_duration_spin.blockSignals(False)
            self.config_start_spin.blockSignals(False)
            self.config_checkpoint_spin.blockSignals(False)
            self.config_energy_edit.blockSignals(False)
            self.config_wind_edit.blockSignals(False)
            self.config_dispatch_combo.blockSignals(False)
            self.config_storage_combo.blockSignals(False)

        self._scenario_manifest = dict(self._scenario_manifest or {})
        self._scenario_manifest["simulation_config_file"] = DEFAULT_SIMULATION_FILE
        self._scenario_manifest = refresh_manifest_file_hashes(self._scenario_manifest)
        self._clear_configuration_dirty()

    def _save_configuration_to_workspace(self):
        workspace_dir = self._get_active_workspace_dir()
        if not workspace_dir:
            QMessageBox.warning(self, "Configuration", "No active imported workspace.")
            return
        if not self._is_copy_manifest():
            QMessageBox.warning(
                self,
                "Configuration",
                "Active scenario is not staged as a copy workspace; save is disabled.",
            )
            return

        sim_data = self._simulation_form_to_dict()
        try:
            model = SimulationConfig.model_validate(sim_data)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Invalid Configuration",
                f"Simulation configuration validation failed:\n{exc}",
            )
            return

        destination = workspace_dir / DEFAULT_SIMULATION_FILE
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                model.model_dump(),
                handle,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        self._scenario_manifest = dict(self._scenario_manifest or {})
        self._scenario_manifest["simulation_config_file"] = DEFAULT_SIMULATION_FILE
        self._scenario_manifest = refresh_manifest_file_hashes(self._scenario_manifest)
        self._clear_configuration_dirty()
        QMessageBox.information(self, "Configuration", "simulation_config.yaml saved to workspace.")

    def _browse_config_data_file(self, field: str):
        workspace_dir = self._get_active_workspace_dir()
        start_dir = str(workspace_dir) if workspace_dir else str(Path(".").resolve())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            start_dir,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not file_path:
            return
        if field == "energy":
            self.config_energy_edit.setText(file_path)
        elif field == "wind":
            self.config_wind_edit.setText(file_path)

    def _select_opex_file_for_workspace(self):
        self._select_economics_file_for_workspace(
            title="Select OPEX Config",
            manifest_key="opex_file",
            canonical_rel=DEFAULT_OPEX_FILE,
            validator=self._validate_opex_text,
        )

    def _select_equipment_file_for_workspace(self):
        self._select_economics_file_for_workspace(
            title="Select Equipment Mapping Config",
            manifest_key="equipment_file",
            canonical_rel=DEFAULT_EQUIPMENT_FILE,
            validator=self._validate_capex_text,
        )

    def _select_economics_file_for_workspace(self, title: str, manifest_key: str, canonical_rel: str, validator):
        workspace_dir = self._get_active_workspace_dir()
        if not workspace_dir:
            QMessageBox.warning(self, "Economics", "No active imported workspace.")
            return
        if not self._is_copy_manifest():
            QMessageBox.warning(
                self,
                "Economics",
                "Active scenario is not staged as a copy workspace; import is disabled.",
            )
            return

        start_dir = str(Path.cwd())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            start_dir,
            "YAML Files (*.yaml *.yml)",
        )
        if not file_path:
            return

        source = Path(file_path)
        try:
            text = load_yaml_text(source)
            validator(text)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Economics",
                f"Selected file is invalid and was not imported:\n{exc}",
            )
            return

        try:
            copy_into_workspace(
                source,
                workspace_dir,
                canonical_rel,
                required=True,
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Economics",
                f"Failed to copy file into workspace:\n{exc}",
            )
            return

        self._scenario_manifest = dict(self._scenario_manifest or {})
        self._scenario_manifest["scenarios_dir"] = str(workspace_dir)
        self._scenario_manifest[manifest_key] = canonical_rel
        self._scenario_manifest = refresh_manifest_file_hashes(self._scenario_manifest)
        self._refresh_economics_tab()

    def _refresh_economics_tab(self):
        self._capex_editing = False
        self._opex_editing = False
        self._capex_valid = False
        self._opex_valid = False

        workspace_dir = self._get_active_workspace_dir()
        if not workspace_dir:
            self.econ_workspace_label.setText("Active workspace: (none)")
            self.opex_status_label.setText("OPEX: not loaded")
            self.equipment_status_label.setText("Equipment mappings: not loaded")
            self.capex_path_label.setText("(missing)")
            self.opex_path_label.setText("(missing)")
            self.capex_editor.setPlainText("No CAPEX file in workspace.")
            self.opex_editor.setPlainText("No OPEX file in workspace.")
            self._capex_original_text = ""
            self._opex_original_text = ""
            self._capex_current_text = ""
            self._opex_current_text = ""
            self._active_capex_path = None
            self._active_opex_path = None
            self._set_general_econ_fields(extract_general_econ_info({}))
            self._set_general_econ_fields_enabled(False)
            self._set_capex_expanded(False)
            self._set_opex_expanded(False)
            self._update_capex_editor_actions()
            self._update_opex_editor_actions()
            return

        self.econ_workspace_label.setText(f"Active workspace: {workspace_dir}")
        self._set_capex_expanded(False)
        self._set_opex_expanded(False)

        opex_path = resolve_manifest_file(self._scenario_manifest, "opex_file", DEFAULT_OPEX_FILE)
        equipment_path = resolve_manifest_file(
            self._scenario_manifest,
            "equipment_file",
            DEFAULT_EQUIPMENT_FILE,
        )
        self._active_opex_path = opex_path
        self._active_capex_path = equipment_path

        if equipment_path and equipment_path.exists():
            self.equipment_status_label.setText(str(equipment_path.name))
            self.capex_path_label.setText(str(equipment_path))
            try:
                capex_text = load_yaml_text(equipment_path)
                capex_data = validate_capex_yaml_text(capex_text)
                self.capex_editor.setPlainText(capex_text)
                self._capex_original_text = capex_text
                self._capex_current_text = capex_text
                self._set_general_econ_fields(extract_general_econ_info(capex_data))
                self._capex_valid = True
            except Exception as exc:
                self.capex_editor.setPlainText(f"Invalid CAPEX YAML:\n{exc}")
                self._capex_original_text = ""
                self._capex_current_text = ""
                self._set_general_econ_fields(extract_general_econ_info({}))
                self._capex_valid = False
        else:
            self.equipment_status_label.setText("(missing)")
            self.capex_path_label.setText("(missing)")
            self.capex_editor.setPlainText("No CAPEX file in workspace.")
            self._capex_original_text = ""
            self._capex_current_text = ""
            self._set_general_econ_fields(extract_general_econ_info({}))
            self._capex_valid = False

        if opex_path and opex_path.exists():
            self.opex_status_label.setText(str(opex_path.name))
            self.opex_path_label.setText(str(opex_path))
            try:
                opex_text = load_yaml_text(opex_path)
                validate_opex_yaml_text(opex_text)
                self.opex_editor.setPlainText(opex_text)
                self._opex_original_text = opex_text
                self._opex_current_text = opex_text
                self._opex_valid = True
            except Exception as exc:
                self.opex_editor.setPlainText(f"Invalid OPEX YAML:\n{exc}")
                self._opex_original_text = ""
                self._opex_current_text = ""
                self._opex_valid = False
        else:
            self.opex_status_label.setText("(missing)")
            self.opex_path_label.setText("(missing)")
            self.opex_editor.setPlainText("No OPEX file in workspace.")
            self._opex_original_text = ""
            self._opex_current_text = ""
            self._opex_valid = False

        self._update_capex_editor_actions()
        self._update_opex_editor_actions()

    def _refresh_workspace_tabs(self):
        self._refresh_economics_tab()
        self._load_configuration_from_workspace()

    def _stage_workspace_from_manifest(self, source_manifest):
        source_manifest = dict(source_manifest or {})
        if not source_manifest.get("scenarios_dir"):
            raise ValueError("Cannot stage workspace without a scenarios_dir source.")
        staged = create_workspace_from_sources(
            source_manifest,
            workspace_root=self._workspace_root,
        )
        self._scenario_manifest = staged
        self._template_source_manifest = dict(staged)
        self._refresh_workspace_tabs()
    
    # ---- FILE OPERATIONS ----
    def new_layout(self):
        """Create a new empty layout."""
        reply = QMessageBox.question(self, "New Layout",
                                     "Clear current layout?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._clear_zone_overlays()
            self._visual_layout = None
            self.graph.clear_session()
            self._restore_topology_analysis(None)
    
    def save_layout(self):
        """Save current layout (template mode: export bundle + .h2plant)."""
        if not hasattr(self, 'current_file') or not self.current_file:
            self.save_layout_as()
        else:
            try:
                if self._template_mode:
                    bundle_dir = Path(self.current_file).with_name(
                        Path(self.current_file).stem + "_bundle"
                    )
                    self._export_bundle_for_save(bundle_dir)
                snapshot = self.persistence_mgr.create_snapshot(self.graph, {})
                snapshot.topology_analysis = self._build_topology_analysis_payload()
                self.persistence_mgr.save(self.current_file, snapshot)
                QMessageBox.information(self, "Success", "Layout saved!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed: {e}")
    
    def save_layout_as(self):
        """Save layout with file dialog (template mode: export bundle alongside)."""
        if self._template_mode and not self.current_file:
            # Auto-generate a slug under layouts/generated/
            from datetime import datetime
            slug = datetime.now().strftime("scenario_%Y%m%d_%H%M%S")
            gen_dir = Path(__file__).resolve().parents[1] / "layouts" / "generated" / slug
            gen_dir.mkdir(parents=True, exist_ok=True)
            default_path = str(gen_dir / f"{slug}.h2plant")
        else:
            default_path = ""

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Layout", default_path, "H2_Plant Files (*.h2plant)"
        )
        if filepath:
            try:
                if self._template_mode:
                    bundle_dir = Path(filepath).parent / f"{Path(filepath).stem}_bundle"
                    self._export_bundle_for_save(bundle_dir)
                snapshot = self.persistence_mgr.create_snapshot(self.graph, {})
                snapshot.topology_analysis = self._build_topology_analysis_payload()
                self.persistence_mgr.save(filepath, snapshot)
                self.current_file = filepath
                QMessageBox.information(self, "Success", "Layout saved!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed: {e}")

    def _export_bundle_for_save(self, bundle_dir: Path):
        """Export scenario bundle and update manifest to point at it."""
        source_manifest = self._template_source_manifest or self._scenario_manifest or {}
        bundle_manifest = export_bundle(
            graph=self.graph,
            template_manifest=source_manifest,
            economics=self._scenario_economics,
            output_dir=bundle_dir,
            scenario_name=source_manifest.get("scenario_name"),
        )
        self._generated_bundle_dir = Path(bundle_manifest["bundle_dir"])
        # Update scenario_manifest to point at generated bundle
        self._scenario_manifest = dict(self._scenario_manifest or {})
        self._scenario_manifest["scenarios_dir"] = str(self._generated_bundle_dir)
        # Clear stale hashes — they referenced original template files
        self._scenario_manifest.pop("file_hashes", None)
        logger.info(f"Bundle exported to {self._generated_bundle_dir}")
    
    def load_layout(self):
        """Load layout from file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Layout", "", "H2_Plant Files (*.h2plant)"
        )
        if filepath:
            try:
                self.graph.clear_session()

                snapshot = self.persistence_mgr.load(filepath)
                self.persistence_mgr.restore_to_graph(self.graph, snapshot)
                self._restore_topology_analysis(snapshot.topology_analysis)
                
                self.current_file = filepath
                QMessageBox.information(self, "Success", "Layout loaded!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Load failed: {e}")

    def open_prebuilt_visual_twin(
        self,
        _checked: bool = False,
        *,
        variant_label: str = "SOEC + PEM + ATR",
        topology_file: str = "plant_topology.yaml",
        equipment_file: str = DEFAULT_EQUIPMENT_FILE,
        prebuilt_filename: str = "plant_topology_visual.h2plant",
        project_name: str = "Plant Topology Visual Twin (SOEC + PEM + ATR)",
    ):
        """Load the committed prebuilt scenario visual twin layout."""
        prebuilt_path = (
            Path(__file__).resolve().parents[1] / "layouts" / prebuilt_filename
        )
        generated_note = ""
        repo_root = Path(__file__).resolve().parents[3]
        scenarios_dir = repo_root / "scenarios"
        force_regenerate = False
        refresh_reason = ""

        if prebuilt_path.exists():
            force_regenerate, refresh_reason = prebuilt_layout_needs_regeneration(
                canonical_path=prebuilt_path,
                scenarios_dir=str(scenarios_dir),
                topology_file=topology_file,
                equipment_file=equipment_file,
            )

        if not prebuilt_path.exists() or force_regenerate:
            try:
                (
                    prebuilt_path,
                    was_generated,
                    used_temp_fallback,
                    node_count,
                    edge_count,
                ) = ensure_prebuilt_layout_file(
                    canonical_path=prebuilt_path,
                    scenarios_dir=str(scenarios_dir),
                    topology_file=topology_file,
                    equipment_file=equipment_file,
                    project_name=project_name,
                    force_regenerate=force_regenerate,
                )
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Prebuilt visual twin not found and auto-generation failed:\n"
                    f"{exc}",
                )
                return

            if was_generated:
                if used_temp_fallback:
                    prefix = "Prebuilt visual twin was regenerated" if force_regenerate else "Prebuilt visual twin was missing and generated"
                    reason = f"\nReason: {refresh_reason}" if force_regenerate and refresh_reason else ""
                    generated_note = (
                        f"{prefix}{reason} to a temporary file:\n"
                        f"{prebuilt_path}\n"
                        f"({node_count} nodes, {edge_count} edges)."
                    )
                    logger.warning(generated_note)
                else:
                    if force_regenerate:
                        reason = f"\nReason: {refresh_reason}" if refresh_reason else ""
                        generated_note = (
                            "Prebuilt visual twin was regenerated from source scenario files"
                            f"{reason}:\n"
                            f"{prebuilt_path}\n"
                            f"({node_count} nodes, {edge_count} edges)."
                        )
                    else:
                        generated_note = (
                            "Prebuilt visual twin was missing and has been generated:\n"
                            f"{prebuilt_path}\n"
                            f"({node_count} nodes, {edge_count} edges)."
                        )
                    logger.info(generated_note)
        elif refresh_reason and refresh_reason != "up_to_date":
            logger.info(f"Prebuilt visual twin refresh check: {refresh_reason}")

        try:
            self.graph.clear_session()
            snapshot = self.persistence_mgr.load(str(prebuilt_path))
            self.persistence_mgr.restore_to_graph(self.graph, snapshot)
            self._restore_topology_analysis(snapshot.topology_analysis)
            # Apply angle pipe routing for prebuilt report-mode layouts
            try:
                from NodeGraphQt.constants import PipeLayoutEnum
                self.graph.set_pipe_style(PipeLayoutEnum.ANGLE)
            except Exception as pipe_exc:
                logger.debug("Could not apply ANGLE pipe layout: %s", pipe_exc)

            repo_scenarios_dir = Path(__file__).resolve().parents[3] / "scenarios"
            fallback_manifest = {
                "scenarios_dir": str(repo_scenarios_dir),
                "topology_file": topology_file,
                "physics_file": "physics_parameters.yaml",
                "economics_file": "economics_parameters.yaml",
                "simulation_config_file": "simulation_config.yaml",
                "equipment_file": equipment_file,
                "opex_file": DEFAULT_OPEX_FILE,
            }
            source_manifest = dict(self._scenario_manifest or {})
            if not source_manifest.get("scenarios_dir"):
                source_manifest = dict(fallback_manifest)
            try:
                self._stage_workspace_from_manifest(source_manifest)
            except Exception as primary_exc:
                if source_manifest.get("scenarios_dir") == str(repo_scenarios_dir):
                    raise
                try:
                    self._stage_workspace_from_manifest(fallback_manifest)
                except Exception as fallback_exc:
                    raise RuntimeError(
                        "Failed to stage workspace from prebuilt manifest and fallback defaults. "
                        f"Primary: {primary_exc}; Fallback: {fallback_exc}"
                    ) from fallback_exc

            # Open as template: avoid accidental overwrite of canonical artifact.
            self.current_file = None
            self._enter_template_mode()

            message = (
                f"Prebuilt visual twin ({variant_label}) loaded as template.\n"
                "Use 'Save Layout As...' to keep a copy."
            )
            if generated_note:
                message = f"{generated_note}\n\n{message}"

            QMessageBox.information(self, "Success", message)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load prebuilt visual twin ({variant_label}): {e}",
            )

    def import_scenario_visual(self):
        """Import scenario YAML files as a visual graph twin."""
        scenarios_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Scenarios Directory",
            str(Path("scenarios").resolve()),
            QFileDialog.ShowDirsOnly,
        )
        if not scenarios_dir:
            return

        topology_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Topology File",
            scenarios_dir,
            "Topology YAML (plant_topology*.yaml);;YAML Files (*.yaml *.yml)",
        )
        if not topology_file:
            return

        if self.graph.all_nodes():
            reply = QMessageBox.question(
                self,
                "Import Scenario Visual",
                "This will clear the current graph before import. Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        try:
            equipment_file = infer_equipment_file_for_topology(topology_file)
            model = ScenarioVisualImporter.build_visual_model(
                scenarios_dir=scenarios_dir,
                topology_file=topology_file,
                equipment_file=equipment_file,
            )

            self.graph.clear_session()
            self._restore_topology_analysis(None)
            node_map = {}

            for visual_node in model.nodes:
                node = self._create_visual_twin_node(visual_node)
                node_map[visual_node.id] = node

            for edge in model.edges:
                src_node = node_map.get(edge.source_id)
                tgt_node = node_map.get(edge.target_id)
                if src_node is None or tgt_node is None:
                    continue
                self._connect_visual_edge(src_node, tgt_node, edge.source_port, edge.target_port)

            # Use angle routing so imported process lines read like orthogonal PFD runs.
            try:
                from NodeGraphQt.constants import PipeLayoutEnum
                self.graph.set_pipe_style(PipeLayoutEnum.ANGLE)
            except Exception as pipe_exc:
                logger.debug("Could not apply ANGLE pipe layout after import: %s", pipe_exc)

            self._restore_topology_analysis(model.metadata)
            self._stage_workspace_from_manifest(self._scenario_manifest or {})
            self.current_file = None
            self._enter_template_mode()
            QMessageBox.information(
                self,
                "Import Complete",
                f"Imported {len(model.nodes)} nodes and {len(model.edges)} connections from scenario.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", f"Scenario visual import failed:\n{exc}")

    def _typed_backend_node_map(self):
        return {
            # Electrolysis / Production
            "PEM": PEMStackNode,
            "SOEC": SOECStackNode,
            "PowerTransformer": RectifierNode,
            # Thermal
            "Chiller": ChillerNode,
            "DryCooler": DryCoolerNode,
            "Interchanger": InterchangerNode,
            "ElectricBoiler": ElectricBoilerNode,
            "Attemperator": AttemperatorNode,
            "CoolingManager": CoolingManagerNode,
            # Separation
            "Coalescer": CoalescerNode,
            "KnockOutDrum": KnockOutDrumNode,
            "PSA Unit": PSAUnitNode,
            "DeoxoReactor": DeoxoReactorNode,
            "HydrogenMultiCyclone": HydrogenMultiCycloneNode,
            "SeparationTank": SeparationTankNode,
            "SyngasPSA": SyngasPSANode,
            # Mixing / Flow
            "Mixer": MixerNode,
            "Valve": ValveNode,
            "StreamSplitter": StreamSplitterNode,
            "DrainRecorderMixer": DrainRecorderMixerNode,
            "SignalMakeupMixer": SignalMakeupMixerNode,
            "ProportionalMakeupMixer": ProportionalMakeupMixerNode,
            "OxygenMakeupNode": OxygenMakeupNode,
            # Water
            "WaterPurifier": WaterPurifierNode,
            "UltraPureWaterTank": UltraPureWaterTankNode,
            "ExternalWaterSource": ExternalWaterSourceNode,
            "WaterPumpThermodynamic": WaterPumpThermodynamicNode,
            # Storage / Delivery
            "DetailedTank": DetailedTankNode,
            "DischargeStation": DischargeStationNode,
            "CompressorSingle": CompressorSingleNode,
            # Reforming
            "IntegratedATRPlant": IntegratedATRPlantNode,
            "ATR_Boiler": ATRBoilerNode,
            "BiogasSource": BiogasSourceNode,
        }

    def _create_visual_twin_node(self, visual_node):
        """Create GUI node for a normalized scenario node record."""
        node_class = self._typed_backend_node_map().get(visual_node.backend_type, ScenarioComponentNode)
        display_name = f"{visual_node.backend_type}: {visual_node.id}"
        node = self.graph.create_node(
            node_class,
            name=display_name,
            pos=[visual_node.x, visual_node.y],
        )

        if isinstance(node, ScenarioComponentNode):
            node.configure_from_scenario(
                component_id=visual_node.id,
                backend_type=visual_node.backend_type,
                input_ports=visual_node.incoming_ports,
                output_ports=visual_node.outgoing_ports,
                params=visual_node.params,
            )
            # Keep explicit compatibility metadata on fallback nodes.
            self._set_hidden_node_property(node, "__scenario_component_id", visual_node.id)
            self._set_hidden_node_property(node, "__scenario_backend_type", visual_node.backend_type)
            self._set_hidden_node_property(node, "__scenario_locked", True)
            node.set_disabled(True)  # Visual read-only indicator for scenario nodes
            self._set_hidden_node_property(node, "__scenario_unmapped_params", {})
        else:
            if "component_id" in node.properties():
                node.set_property("component_id", visual_node.id)
            unmapped = self._apply_params_to_typed_node(node, visual_node.backend_type, visual_node.params)
            self._ensure_node_ports(node, visual_node.incoming_ports, visual_node.outgoing_ports)
            # Typed nodes also need backend_type for export identity resolution
            self._set_hidden_node_property(node, "__scenario_backend_type", visual_node.backend_type)
            self._set_hidden_node_property(node, "__scenario_unmapped_params", dict(unmapped))

        # Typed nodes persist only minimal scenario metadata needed for port
        # restoration and selection-linked equipment lookup.
        self._set_hidden_node_property(node, "__scenario_inputs", list(visual_node.incoming_ports))
        self._set_hidden_node_property(node, "__scenario_outputs", list(visual_node.outgoing_ports))
        self._set_hidden_node_property(node, "__scenario_params", dict(visual_node.params))

        return node

    def _apply_params_to_typed_node(self, node, backend_type: str, params: dict):
        """Apply canonical backend params to typed node properties."""
        node_props = node.properties()
        mapped_props, unmapped = backend_to_gui_props(
            backend_type=backend_type,
            backend_params=dict(params or {}),
            available_props=set(node_props.keys()),
        )
        for key, value in mapped_props.items():
            if key == "component_id" and "component_id" in node_props:
                continue
            if key not in node_props:
                continue
            if isinstance(value, (int, float, bool)):
                node.set_property(key, str(value))
            else:
                node.set_property(key, value)
        return unmapped

    def _ensure_node_ports(self, node, input_ports, output_ports):
        """Ensure node has exact scenario port names required for imported edges."""
        existing_inputs = {port.name() for port in node.input_ports()}
        existing_outputs = {port.name() for port in node.output_ports()}

        for port_name in input_ports:
            if port_name in existing_inputs:
                continue
            try:
                node.add_input(port_name, flow_type=_infer_flow_type(port_name).value, multi_input=True)
            except TypeError:
                node.add_input(port_name)

        for port_name in output_ports:
            if port_name in existing_outputs:
                continue
            try:
                node.add_output(port_name, flow_type=_infer_flow_type(port_name).value, multi_output=True)
            except TypeError:
                node.add_output(port_name)

    def _connect_visual_edge(self, src_node, tgt_node, source_port: str, target_port: str):
        """Connect imported ports, creating any missing endpoint ports defensively."""
        src_port_obj = src_node.get_output(source_port)
        if src_port_obj is None:
            self._ensure_node_ports(src_node, [], [source_port])
            src_port_obj = src_node.get_output(source_port)

        tgt_port_obj = tgt_node.get_input(target_port)
        if tgt_port_obj is None:
            self._ensure_node_ports(tgt_node, [target_port], [])
            tgt_port_obj = tgt_node.get_input(target_port)

        if src_port_obj and tgt_port_obj:
            src_port_obj.connect_to(tgt_port_obj)

    def _set_hidden_node_property(self, node, key: str, value):
        """Create/update hidden property on a node for persistence metadata."""
        props = node.properties()
        if key not in props and hasattr(node, "create_property"):
            node.create_property(key, value=value, widget_type=0)
        else:
            node.set_property(key, value)

    def _enter_template_mode(self):
        """Activate template authoring mode — nodes become editable."""
        self._template_mode = True
        self._template_source_manifest = dict(self._scenario_manifest) if self._scenario_manifest else None
        self._generated_bundle_dir = None
        # Re-enable scenario nodes so user can edit params
        for node in self.graph.all_nodes():
            node.set_disabled(False)
        # Make economics table editable
        self.scenario_economics_table.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.scenario_economics_table.setSelectionMode(QAbstractItemView.SingleSelection)
        if not self._economics_editing_connected:
            self.scenario_economics_table.cellChanged.connect(self._on_economics_cell_changed)
            self._economics_editing_connected = True
        self._populate_scenario_economics_table()  # re-populate with editable flags
        logger.info("Template authoring mode activated")

    def _exit_template_mode(self):
        """Deactivate template mode (normal graph mode)."""
        self._template_mode = False
        self._template_source_manifest = None
        self._generated_bundle_dir = None
        # Restore read-only economics table
        self.scenario_economics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.scenario_economics_table.setSelectionMode(QAbstractItemView.NoSelection)
        if self._economics_editing_connected:
            try:
                self.scenario_economics_table.cellChanged.disconnect(self._on_economics_cell_changed)
            except RuntimeError:
                pass  # not connected
        self._economics_editing_connected = False

    # ---- ZONE OVERLAY RENDERING ----

    def _draw_zone_overlays(self, visual_layout: dict) -> None:
        """Draw decorative swimlane rectangles for each process zone on the graph scene.

        Overlays are fixed-position QGraphicsRectItem items placed behind all nodes.
        They reflect the initial import positions and are not updated when nodes are dragged.
        """
        from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsTextItem
        from PySide6.QtGui import QColor, QPen, QFont
        from PySide6.QtCore import Qt

        self._clear_zone_overlays()

        scene = self.graph.scene()
        if scene is None:
            return

        zones = visual_layout.get("zones", {})
        for zone_name, zone_rect in zones.items():
            x = float(zone_rect.get("x", 0.0))
            y = float(zone_rect.get("y", 0.0))
            w = float(zone_rect.get("w", 200.0))
            h = float(zone_rect.get("h", 200.0))
            color_list = zone_rect.get("color", [200, 200, 200])
            r, g, b = int(color_list[0]), int(color_list[1]), int(color_list[2])

            rect_item = QGraphicsRectItem(x, y, w, h)
            bg_color = QColor(r, g, b, 60)   # semi-transparent fill
            border_color = QColor(r, g, b, 160)
            rect_item.setBrush(bg_color)
            rect_item.setPen(QPen(border_color, 2))
            rect_item.setZValue(-100)  # behind all nodes
            scene.addItem(rect_item)
            self._zone_overlay_items.append(rect_item)

            label_item = QGraphicsTextItem(zone_name)
            font = QFont()
            font.setBold(True)
            font.setPointSize(9)
            label_item.setFont(font)
            label_item.setDefaultTextColor(QColor(r // 2, g // 2, b // 2, 220))
            label_item.setPos(x + 8, y + 4)
            label_item.setZValue(-99)
            scene.addItem(label_item)
            self._zone_overlay_items.append(label_item)

        logger.debug("Drew %d zone overlays", len(zones))

    def _clear_zone_overlays(self) -> None:
        """Remove all zone overlay items from the graph scene."""
        scene = self.graph.scene() if hasattr(self, "graph") else None
        for item in self._zone_overlay_items:
            try:
                if scene is not None:
                    scene.removeItem(item)
            except Exception as exc:
                logger.debug("Failed to remove zone overlay item: %s", exc)
        self._zone_overlay_items = []

    def _apply_zone_colors(self, visual_layout: dict) -> None:
        """Apply zone-based background colors to nodes after they are placed in the graph."""
        from h2_plant.gui.core.visual_layout_policy import BLOCK_COLORS, ZONE_COLORS
        node_zone_map = visual_layout.get("node_zone_map", {})
        zones_meta = visual_layout.get("zones", {})
        for node in self.graph.all_nodes():
            node_id = node.name() if hasattr(node, "name") else str(node)
            # Try component_id property first (canonical ID)
            try:
                node_id = node.get_property("component_id") or node_id
            except Exception as exc:
                logger.debug("Could not read component_id from node '%s': %s", node, exc)
            zone = node_zone_map.get(node_id)
            if zone:
                color = ZONE_COLORS.get(zone) or BLOCK_COLORS.get(zone)
                if color is None and isinstance(zones_meta.get(zone), dict):
                    c = zones_meta[zone].get("color")
                    if isinstance(c, list) and len(c) >= 3:
                        color = (int(c[0]), int(c[1]), int(c[2]))
                if color:
                    r, g, b = color
                    try:
                        node.set_color(r, g, b)
                    except Exception as exc:
                        logger.debug("Could not apply zone color to node '%s': %s", node_id, exc)

    def _run_auto_layout_report_mode(self) -> None:
        """Recalculate industrial PFD layout for the current scenario and redraw overlays.

        Only works when a scenario manifest is active (i.e., a scenario has been imported).
        Node positions are updated; overlays are redrawn from fresh layout data.
        """
        manifest = self._scenario_manifest
        if not manifest or not manifest.get("scenarios_dir"):
            QMessageBox.information(
                self,
                "Auto-Layout",
                "No active scenario. Open a prebuilt visual twin or import a scenario first.",
            )
            return

        scenarios_dir = manifest.get("scenarios_dir", "")
        topology_file = manifest.get("topology_file", "plant_topology.yaml")
        equipment_file = manifest.get("equipment_file", DEFAULT_EQUIPMENT_FILE)
        # topology_file may be relative to scenarios_dir
        from pathlib import Path as _Path
        tp = _Path(topology_file)
        if not tp.is_absolute():
            topology_file = str(_Path(scenarios_dir) / tp)

        try:
            from h2_plant.gui.core.scenario_visual_importer import ScenarioVisualImporter as _SVI
            model = _SVI.build_visual_model(
                scenarios_dir=scenarios_dir,
                topology_file=topology_file,
                layout_mode="report_v1",
                equipment_file=equipment_file,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Auto-Layout Error", f"Industrial layout calculation failed:\n{exc}")
            return

        # Apply new positions to live graph nodes
        from NodeGraphQt.constants import PipeLayoutEnum
        node_id_to_node = {
            (n.get_property("component_id") or n.name()): n
            for n in self.graph.all_nodes()
        }
        for visual_node in model.nodes:
            live_node = node_id_to_node.get(visual_node.id)
            if live_node is not None:
                live_node.set_pos(visual_node.x, visual_node.y)

        # Set angle pipe routing
        try:
            self.graph.set_pipe_style(PipeLayoutEnum.ANGLE)
        except Exception as exc:
            logger.debug("Could not apply ANGLE pipe style for auto-layout: %s", exc)

        # Redraw overlays
        vl = model.metadata.get("visual_layout")
        if vl:
            self._visual_layout = dict(vl)
            self._draw_zone_overlays(self._visual_layout)
            self._apply_zone_colors(self._visual_layout)

        self.graph.fit_to_selection()

    # ---- TOPOLOGY ANALYSIS PAYLOAD ----

    def _build_topology_analysis_payload(self):
        """Collect additive snapshot metadata for scenario-imported layouts."""
        if not self._scenario_manifest:
            return None
        payload = {
            "scenario_manifest": dict(self._scenario_manifest),
            "scenario_economics": dict(self._scenario_economics),
            "scenario_equipment_entries": list(self._scenario_equipment_entries),
            "scenario_equipment_index": dict(self._scenario_equipment_index),
        }
        if self._template_mode:
            payload["template_mode"] = True
        if self._generated_bundle_dir:
            payload["generated_bundle_manifest"] = {
                "bundle_dir": str(self._generated_bundle_dir),
            }
        if self._visual_layout is not None:
            payload["visual_layout"] = dict(self._visual_layout)
        return payload

    def _check_manifest_hash_drift(self):
        """Warn if scenario YAML files have changed since the manifest was saved."""
        manifest = self._scenario_manifest
        if not manifest:
            return
        stored_hashes = manifest.get("file_hashes", {})
        if not stored_hashes:
            return
        scenarios_dir = Path(manifest.get("scenarios_dir", ""))
        drifted = []
        for name, expected_hash in stored_hashes.items():
            # v2 uses basenames, v1 used absolute paths
            candidate = scenarios_dir / name if not Path(name).is_absolute() else Path(name)
            if not candidate.exists():
                continue
            try:
                from h2_plant.gui.core.scenario_visual_importer import _sha256_file
                actual = _sha256_file(candidate)
                if actual != expected_hash:
                    drifted.append(candidate.name)
            except Exception as exc:
                logger.warning("Failed to compare hash for '%s': %s", candidate, exc)
        if drifted:
            logger.warning(f"Scenario hash drift: {', '.join(drifted)}")
            QMessageBox.warning(
                self,
                "Scenario File Changed",
                f"The following scenario files have changed since "
                f"this project was saved:\n\n• {'  • '.join(drifted)}\n\n"
                f"Re-import the scenario to pick up the latest changes.",
            )

    def _restore_topology_analysis(self, topology_analysis):
        """Restore scenario metadata from snapshot payload."""
        # Always clear existing overlays before (re)loading.
        self._clear_zone_overlays()
        self._visual_layout = None

        if not topology_analysis or not topology_analysis.get("scenario_manifest"):
            self._scenario_manifest = None
            self._scenario_economics = {}
            self._scenario_equipment_entries = []
            self._scenario_equipment_index = {}
            self._last_selected_component_id = None
            self._exit_template_mode()
            self._populate_scenario_economics_table()
            self._populate_equipment_mapping_table([])
            self._populate_imported_params_table({}, no_selection=True)
            self.scenario_economics_dock.hide()
            self.equipment_mapping_dock.hide()
            self.imported_params_dock.hide()
            self._refresh_workspace_tabs()
            return

        self._scenario_manifest = dict(topology_analysis.get("scenario_manifest") or {})
        self._check_manifest_hash_drift()
        self._scenario_economics = dict(
            topology_analysis.get("scenario_economics")
            or topology_analysis.get("economics")
            or {}
        )
        self._scenario_equipment_entries = list(
            topology_analysis.get("scenario_equipment_entries")
            or topology_analysis.get("equipment_entries")
            or []
        )
        self._scenario_equipment_index = dict(
            topology_analysis.get("scenario_equipment_index")
            or topology_analysis.get("equipment_index")
            or {}
        )
        self._last_selected_component_id = None

        self._populate_scenario_economics_table()
        self._refresh_equipment_panel_selection(force=True)
        self.scenario_economics_dock.show()
        self.equipment_mapping_dock.show()
        self.imported_params_dock.show()
        self._refresh_workspace_tabs()

        # Restore template authoring mode from persisted flag
        if topology_analysis.get("template_mode"):
            self._enter_template_mode()
            bundle_meta = topology_analysis.get("generated_bundle_manifest")
            if bundle_meta and bundle_meta.get("bundle_dir"):
                self._generated_bundle_dir = Path(bundle_meta["bundle_dir"])

        # Restore zone overlay from persisted visual_layout (present only in report_v1 layouts)
        vl = topology_analysis.get("visual_layout")
        if vl and isinstance(vl, dict) and vl.get("zones"):
            self._visual_layout = dict(vl)
            try:
                self._draw_zone_overlays(self._visual_layout)
                self._apply_zone_colors(self._visual_layout)
            except Exception as overlay_exc:
                logger.warning("Zone overlay rendering failed (non-fatal): %s", overlay_exc)

    def _populate_scenario_economics_table(self):
        economics = self._scenario_economics or {}
        self.scenario_economics_table.blockSignals(True)  # prevent cellChanged during population
        self.scenario_economics_table.setRowCount(0)

        if not economics:
            self.scenario_economics_table.setRowCount(1)
            self.scenario_economics_table.setItem(0, 0, QTableWidgetItem("No imported scenario metadata"))
            self.scenario_economics_table.setItem(0, 1, QTableWidgetItem(""))
            self.scenario_economics_table.blockSignals(False)
            return

        for row, key in enumerate(sorted(economics.keys())):
            self.scenario_economics_table.insertRow(row)
            key_item = QTableWidgetItem(str(key))
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)  # key always read-only
            self.scenario_economics_table.setItem(row, 0, key_item)
            val_item = QTableWidgetItem(str(economics.get(key)))
            if self._template_mode:
                val_item.setFlags(val_item.flags() | Qt.ItemIsEditable)
            else:
                val_item.setFlags(val_item.flags() & ~Qt.ItemIsEditable)
            self.scenario_economics_table.setItem(row, 1, val_item)
        self.scenario_economics_table.blockSignals(False)

    def _on_economics_cell_changed(self, row: int, col: int):
        """Sync edited economics values back to _scenario_economics."""
        if col != 1 or not self._template_mode:
            return
        key_item = self.scenario_economics_table.item(row, 0)
        val_item = self.scenario_economics_table.item(row, 1)
        if not key_item or not val_item:
            return
        key = key_item.text()
        raw_value = val_item.text()
        # Parse with YAML scalar rules to preserve numeric/bool types
        import yaml
        try:
            parsed = yaml.safe_load(raw_value)
        except Exception:
            parsed = raw_value
        # Validate against EconomicsConfig schema — revert cell on failure
        field_info = EconomicsConfig.model_fields.get(key)
        if field_info is not None:
            try:
                EconomicsConfig.model_validate(
                    {**self._scenario_economics, key: parsed}
                )
            except Exception as exc:
                logger.warning(f"Economics '{key}' rejected: {exc}")
                self.scenario_economics_table.blockSignals(True)
                prev = self._scenario_economics.get(key, "")
                val_item.setText(str(prev))
                self.scenario_economics_table.blockSignals(False)
                return
        self._scenario_economics[key] = parsed
        logger.info(f"Economics '{key}' updated to {parsed!r} ({type(parsed).__name__})")

    def _refresh_equipment_panel_selection(self, force: bool = False):
        if not self._scenario_manifest:
            return

        selected = self.graph.selected_nodes()
        if not selected:
            if not force and self._last_selected_component_id is None:
                return
            self._last_selected_component_id = None
            self._populate_equipment_mapping_table([], no_selection=True)
            self._populate_imported_params_table({}, no_selection=True)
            return

        component_id = self._resolve_selected_component_id(selected[0])

        if not force and component_id == self._last_selected_component_id:
            return

        self._last_selected_component_id = component_id
        equipment_indices = self._scenario_equipment_index.get(component_id or "", [])
        entries = [self._scenario_equipment_entries[idx] for idx in equipment_indices if idx < len(self._scenario_equipment_entries)]
        self._populate_equipment_mapping_table(entries, no_selection=False)
        self._populate_imported_params_table(
            self._resolve_selected_unmapped_params(selected[0]),
            no_selection=False,
        )

    def _resolve_selected_component_id(self, node):
        props = node.get_properties() if hasattr(node, "get_properties") else node.properties()
        return resolve_component_id_for_equipment(
            component_id=props.get("component_id"),
            legacy_component_id=props.get("__scenario_component_id"),
            node_name=node.name(),
        )

    def _populate_equipment_mapping_table(self, entries, no_selection: bool = False):
        self.equipment_mapping_table.setRowCount(0)

        if not entries:
            self.equipment_mapping_table.setRowCount(1)
            message = (
                "Select a node to view linked equipment mappings."
                if no_selection
                else "No equipment mapping for selected node."
            )
            self.equipment_mapping_table.setItem(0, 0, QTableWidgetItem(message))
            for col in range(1, 7):
                self.equipment_mapping_table.setItem(0, col, QTableWidgetItem(""))
            return

        for row, entry in enumerate(entries):
            self.equipment_mapping_table.insertRow(row)
            topology_ids = ", ".join(entry.get("topology_ids", []))
            self.equipment_mapping_table.setItem(row, 0, QTableWidgetItem(str(entry.get("tag", ""))))
            self.equipment_mapping_table.setItem(row, 1, QTableWidgetItem(str(entry.get("block", ""))))
            self.equipment_mapping_table.setItem(row, 2, QTableWidgetItem(str(entry.get("name", ""))))
            self.equipment_mapping_table.setItem(row, 3, QTableWidgetItem(str(entry.get("capacity_variable", ""))))
            self.equipment_mapping_table.setItem(row, 4, QTableWidgetItem(str(entry.get("capacity_unit", ""))))
            self.equipment_mapping_table.setItem(row, 5, QTableWidgetItem(str(entry.get("cost_source", ""))))
            self.equipment_mapping_table.setItem(row, 6, QTableWidgetItem(topology_ids))

    def _resolve_selected_unmapped_params(self, node):
        props = node.get_properties() if hasattr(node, "get_properties") else node.properties()
        existing = props.get("__scenario_unmapped_params")
        if isinstance(existing, dict):
            return dict(existing)

        backend_type = str(
            props.get("__scenario_backend_type")
            or props.get("backend_type")
            or ""
        ).strip()
        base_params = props.get("__scenario_params")
        if not backend_type or not isinstance(base_params, dict):
            return {}

        _, unmapped = backend_to_gui_props(
            backend_type=backend_type,
            backend_params=base_params,
            available_props=set(props.keys()),
        )
        self._set_hidden_node_property(node, "__scenario_unmapped_params", dict(unmapped))
        return unmapped

    def _populate_imported_params_table(self, unmapped_params, no_selection: bool = False):
        self.imported_params_table.setRowCount(0)

        if not self._scenario_manifest:
            self.imported_params_dock.hide()
            return

        if not unmapped_params:
            message = (
                "Select a node to inspect unmapped imported parameters."
                if no_selection
                else "No unmapped imported parameters for selected node."
            )
            self.imported_params_table.setRowCount(1)
            self.imported_params_table.setItem(0, 0, QTableWidgetItem(message))
            self.imported_params_table.setItem(0, 1, QTableWidgetItem(""))
            self.imported_params_dock.hide()
            return

        row = 0
        for key in sorted(unmapped_params.keys()):
            self.imported_params_table.insertRow(row)
            self.imported_params_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.imported_params_table.setItem(row, 1, QTableWidgetItem(str(unmapped_params.get(key))))
            row += 1

        self.imported_params_dock.show()
    
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
                logger.error(f"Error deleting node: {e}")
    
    def duplicate_selection(self):
        """Duplicate selected nodes."""
        selected = self.graph.selected_nodes()
        if not selected:
            return

        offset = 50
        new_nodes = []
        skipped_props = {"id", "name", "pos"}

        for node in selected:
            try:
                new_node = self.graph.create_node(
                    node.__class__,
                    name=f"{node.name()}_copy",
                    pos=[node.x_pos() + offset, node.y_pos() + offset],
                )

                source_props = (
                    node.get_properties()
                    if hasattr(node, "get_properties")
                    else node.properties()
                )
                for prop_name, prop_value in dict(source_props).items():
                    if prop_name in skipped_props or prop_name.startswith("__builtin__"):
                        continue
                    try:
                        new_node.set_property(prop_name, copy.deepcopy(prop_value))
                    except Exception as prop_exc:
                        logger.debug(
                            "Skipping duplicate property '%s' for node '%s': %s",
                            prop_name,
                            node.name(),
                            prop_exc,
                        )

                new_nodes.append(new_node)
            except Exception as e:
                logger.error("Error duplicating node '%s': %s", node.name(), e)

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
            logger.error(f"Validation error: {e}")
    
    def toggle_auto_validation(self, enabled):
        """Toggle automatic validation."""
        if enabled:
            self.validation_timer.start(2000)
        else:
            self.validation_timer.stop()

    def _stop_active_simulation_worker(self, wait_ms: int = 3000) -> bool:
        """Request worker stop and wait for completion if currently running."""
        worker = self.worker
        if worker is None:
            return True

        if worker.isRunning():
            logger.info("Stopping active simulation worker.")
            worker.stop()
            if not worker.wait(wait_ms):
                logger.warning("Simulation worker did not stop within %sms.", wait_ms)
                return False

        self.worker = None
        return True
    
    # ---- SIMULATION ----
    def run_simulation(self):
        """Run plant simulation using new backend architecture.
        
        Supports three modes:
        - Template mode: exports current graph to a bundle, runs from generated YAML
        - Scenario mode: loads context from scenarios/ directory via ConfigLoader
        - Graph mode: builds context from canvas nodes via GraphToConfigAdapter
        """
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self,
                "Simulation in Progress",
                "A simulation is already running. Wait for it to finish or cancel it first.",
            )
            return

        try:
            # 0. Configure Simulation
            dialog = SimulationConfigDialog(self)
            if dialog.exec() != QDialog.Accepted:
                return
            
            duration_hours = dialog.get_duration_hours()
            strategy = dialog.get_strategy()
            storage_mode = dialog.get_storage_control_mode()
            requested_scenarios_dir = dialog.get_scenarios_dir()
            override_enabled = dialog.is_override_enabled()

            if self._template_mode:
                # === TEMPLATE MODE: export-before-run ===
                if not self._generated_bundle_dir:
                    from datetime import datetime
                    slug = datetime.now().strftime("run_%Y%m%d_%H%M%S")
                    bundle_dir = Path(__file__).resolve().parents[1] / "layouts" / "generated" / slug
                else:
                    bundle_dir = self._generated_bundle_dir
                source_manifest = self._template_source_manifest or self._scenario_manifest or {}
                bundle_manifest = export_bundle(
                    graph=self.graph,
                    template_manifest=source_manifest,
                    economics=self._scenario_economics,
                    output_dir=bundle_dir,
                    scenario_name=source_manifest.get("scenario_name"),
                )
                self._generated_bundle_dir = Path(bundle_manifest["bundle_dir"])
                scenarios_dir = str(self._generated_bundle_dir)
                topology_file = None
                logger.info(f"Template mode: running from generated bundle at {scenarios_dir}")
            else:
                scenarios_dir, topology_file, _forced_scenario_mode = resolve_simulation_source(
                    self._scenario_manifest,
                    requested_scenarios_dir,
                )

            if scenarios_dir:
                # === SCENARIO MODE ===
                from h2_plant.config.loader import ConfigLoader
                loader = ConfigLoader(scenarios_dir)
                if topology_file:
                    context = loader.load_context(topology_file=topology_file)
                else:
                    context = loader.load_context()
                
                if override_enabled:
                    context.simulation.duration_hours = duration_hours
                    if hasattr(context.simulation, 'dispatch_strategy'):
                        context.simulation.dispatch_strategy = strategy
                    if hasattr(context.simulation, 'storage_control_mode'):
                        context.simulation.storage_control_mode = storage_mode
            else:
                # === GRAPH MODE ===
                adapter = GraphToConfigAdapter()
                
                # Extract Nodes with flow-type-aware ports
                for node in self.graph.all_nodes():
                    ports = []
                    for p in node.input_ports():
                        ports.append(Port(
                            name=p.name(),
                            flow_type=_infer_flow_type(p.name()),
                            direction="input"
                        ))
                    for p in node.output_ports():
                        ports.append(Port(
                            name=p.name(),
                            flow_type=_infer_flow_type(p.name()),
                            direction="output"
                        ))

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
                    
                # Extract Edges with inferred flow types
                for node in self.graph.all_nodes():
                    for output_port in node.output_ports():
                        for target_port in output_port.connected_ports():
                            target_node = target_port.node()
                            
                            edge = GraphEdge(
                                source_node_id=node.id,
                                source_port=output_port.name(),
                                target_node_id=target_node.id,
                                target_port=target_port.name(),
                                flow_type=_infer_flow_type(output_port.name())
                            )
                            adapter.add_edge(edge)
                
                # Generate Context from graph
                try:
                    context = adapter.to_simulation_context()
                except ValueError as e:
                    QMessageBox.critical(
                        self,
                        "Graph Configuration Error",
                        f"Invalid graph configuration:\n{e}"
                    )
                    return
                context.simulation.duration_hours = duration_hours
                if hasattr(context.simulation, 'dispatch_strategy'):
                    context.simulation.dispatch_strategy = strategy
                if hasattr(context.simulation, 'storage_control_mode'):
                    context.simulation.storage_control_mode = storage_mode
            
            # Create Worker with strategy override and scenarios dir
            strategy_override = strategy if (override_enabled or not scenarios_dir) else None
            self.worker = SimulationWorker(
                context,
                strategy_override=strategy_override,
                scenarios_dir=scenarios_dir
            )
            
            # Setup Progress Dialog with Cancel button
            progress = QProgressDialog("Running simulation... (0%)", "Cancel", 0, 100, self)
            progress.setWindowTitle("Simulation in Progress")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setAutoClose(False)
            progress.setAutoReset(False)
            progress.setValue(0)
            self._progress_dialog = progress
            cancel_requested = {"value": False}

            def _disconnect_worker_signals():
                worker = self.worker
                if worker is None:
                    return
                for signal, callback in (
                    (worker.progress, on_progress),
                    (worker.finished, on_finished),
                    (worker.canceled, on_canceled),
                    (worker.error, on_error),
                ):
                    try:
                        signal.disconnect(callback)
                    except (TypeError, RuntimeError) as disconnect_exc:
                        logger.debug("Signal disconnect skipped: %s", disconnect_exc)

            def _finalize_worker_state():
                _disconnect_worker_signals()
                self.worker = None

            def on_cancel():
                if self.worker:
                    logger.info("User requested simulation cancellation")
                    cancel_requested["value"] = True
                    progress.setLabelText("Canceling simulation...")
                    progress.setCancelButton(None)
                    self.worker.stop()

            def on_progress(value: int):
                bounded = max(0, min(100, int(value)))
                progress.setValue(bounded)
                if cancel_requested["value"]:
                    progress.setLabelText(f"Canceling simulation... ({bounded}%)")
                else:
                    progress.setLabelText(f"Running simulation... ({bounded}%)")

            def on_finished(history, registry):
                progress.setValue(100)
                progress.accept()
                
                # Store registry for legacy reports
                self.last_registry = registry
                
                # Pass simulation data directly to report widget (no disk I/O)
                if hasattr(self, 'report_widget'):
                    if scenarios_dir:
                        effective_output_dir = Path(scenarios_dir) / "simulation_output"
                        effective_scenarios_dir = Path(scenarios_dir)
                    else:
                        effective_output_dir = Path("simulation_output")
                        effective_scenarios_dir = None
                    self.report_widget.set_results_source(
                        output_dir=effective_output_dir,
                        scenarios_dir=effective_scenarios_dir,
                    )
                    self.report_widget.set_simulation_data(history)
                    self.central_tabs.setCurrentWidget(self.report_widget)
                
                QMessageBox.information(self, "Simulation", "Simulation completed successfully!")
                _finalize_worker_state()
                if self._progress_dialog is progress:
                    self._progress_dialog = None
                progress.deleteLater()
                
            def on_canceled():
                progress.reject()
                QMessageBox.information(self, "Simulation", "Simulation canceled.")
                _finalize_worker_state()
                if self._progress_dialog is progress:
                    self._progress_dialog = None
                progress.deleteLater()

            def on_error(err_msg):
                progress.reject()
                QMessageBox.critical(self, "Simulation Error", f"Failed to run simulation: {err_msg}")
                _finalize_worker_state()
                if self._progress_dialog is progress:
                    self._progress_dialog = None
                progress.deleteLater()
                
            progress.canceled.connect(on_cancel)
            self.worker.progress.connect(on_progress)
            self.worker.finished.connect(on_finished)
            self.worker.canceled.connect(on_canceled)
            self.worker.error.connect(on_error)
            
            self.worker.start()
            progress.exec_()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Simulation Error", f"Failed to start simulation: {e}")
