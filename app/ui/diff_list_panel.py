"""The findings list (SRS FR-7.1/7.2/7.5, Architecture §3.1 "Diff List Panel").

Shows each clustered difference with what it is (its category, in the category's own colour, and
what happened to it) and gives each one an Ignore control. Ignoring is a soft state on the finding
(`DiffCluster.status`), never a deletion, so what a reviewer dismissed stays
on the record and can be restored.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.engine.aggregation.aggregator import merge_clusters
from app.models.categories import CATEGORY_LABELS, CATEGORY_ORDER, SUBTYPE_LABELS, category_colour
from app.models.diff import DiffCluster
from app.models.result import AlignmentResult
from app.ui.theme import set_variant

logger = logging.getLogger(__name__)

_ALL = "all"


class FindingCard(QFrame):
    """One finding: a bar in its category's colour, what it is, and an Ignore/Restore button."""

    selectRequested = Signal(int)
    ignoreToggled = Signal(int)

    def __init__(self, cluster: DiffCluster, selected: bool, parent=None):
        super().__init__(parent)
        self.cluster_id = cluster.id
        self.setProperty("role", "card")
        self.setProperty("selected", "true" if selected else "false")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        bar = QFrame(self)
        bar.setFixedWidth(5)
        bar.setStyleSheet(f"background: {category_colour(cluster.category)}; border-radius: 2px;")
        self.colour_bar = bar

        title = QLabel(
            f"#{cluster.id}  ·  {CATEGORY_LABELS.get(cluster.category, cluster.category)}  ·  "
            f"{SUBTYPE_LABELS[cluster.subtype].capitalize()}"
            + ("  ·  possible misread" if cluster.uncertain else "")
        )
        title.setStyleSheet("font-weight: 700;")
        detail = QLabel(cluster.detail)
        detail.setWordWrap(True)
        detail.setProperty("secondary", "true")
        self.title_label = title
        self.detail_label = detail

        self.ignore_button = QPushButton("Restore" if cluster.is_ignored else "Ignore")
        self.ignore_button.setFixedWidth(72)
        self.ignore_button.clicked.connect(lambda: self.ignoreToggled.emit(self.cluster_id))

        text = QVBoxLayout()
        text.setSpacing(2)
        text.addWidget(title)
        text.addWidget(detail)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)
        row.addWidget(bar)
        row.addLayout(text, stretch=1)
        row.addWidget(self.ignore_button, alignment=Qt.AlignmentFlag.AlignTop)

        if cluster.is_ignored:
            for label in (title, detail):
                label.setStyleSheet("color: #a3a8b3;")

    def mousePressEvent(self, event) -> None:
        self.selectRequested.emit(self.cluster_id)
        super().mousePressEvent(event)


class DiffListPanel(QWidget):
    clusterSelected = Signal(int)
    visibilityChanged = Signal()  # filters, ignoring or merging changed what should be shown

    def __init__(self, parent=None):
        super().__init__(parent)
        self._clusters: list[DiffCluster] = []
        self._selected_id: int | None = None
        self._merges: list[tuple[DiffCluster, list[DiffCluster], int]] = []  # (merged, the findings it replaced, where)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QHBoxLayout()
        self.title_label = QLabel("Findings (0)")
        self.title_label.setProperty("role", "heading")
        self.title_label.setStyleSheet("font-size: 17px;")
        self.alignment_pill = QLabel("")
        self.alignment_pill.setProperty("pill", "empty")
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.alignment_pill)
        layout.addLayout(header)

        self.summary_label = QLabel("")
        self.summary_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.summary_label)

        self.warnings_label = QLabel("")
        self.warnings_label.setProperty("status", "warn")
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setVisible(False)
        layout.addWidget(self.warnings_label)

        filters = QHBoxLayout()
        self.category_combo = QComboBox()
        self.category_combo.addItem("All kinds", _ALL)
        for key in CATEGORY_ORDER:
            self.category_combo.addItem(CATEGORY_LABELS[key], key)
        filters.addWidget(self.category_combo, stretch=1)
        layout.addLayout(filters)

        checks = QHBoxLayout()
        self.show_ignored_check = QCheckBox("Show ignored")
        # Where the words differ but the pixels say the same ink is there, the reader most likely misread: those are
        # listed apart, on request, so a real change in small print is never hidden and the list is not full of them.
        self.show_uncertain_check = QCheckBox("Show possible misreads")
        self.show_uncertain_check.setToolTip(
            "Text the reader saw differently although the pixels match: most likely a misreading, not a change."
        )
        checks.addWidget(self.show_ignored_check)
        checks.addWidget(self.show_uncertain_check)
        checks.addStretch(1)
        layout.addLayout(checks)

        self._cards_layout = QVBoxLayout()
        self._cards_layout.setSpacing(8)
        self._cards_layout.setContentsMargins(0, 0, 4, 0)
        container = QWidget()
        container.setLayout(self._cards_layout)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setWidget(container)
        layout.addWidget(self._scroll, stretch=1)

        self.category_combo.currentIndexChanged.connect(self._on_filters_changed)
        self.show_ignored_check.toggled.connect(self._on_filters_changed)
        self.show_uncertain_check.toggled.connect(self._on_filters_changed)

    # -- data ---------------------------------------------------------------
    def set_result(
        self,
        clusters: list[DiffCluster],
        alignment: AlignmentResult,
        warnings: list[str],
    ) -> None:
        self._clusters = list(clusters)
        self._selected_id = None
        self._merges = []

        low = alignment.low_confidence
        set_variant(self.alignment_pill, "pill", "warn" if low else "loaded")
        self.alignment_pill.setText(
            f"Alignment {alignment.residual_error_px:.2f} px" + (" · LOW CONFIDENCE" if low else "")
        )

        self.warnings_label.setText("\n".join(warnings))
        self.warnings_label.setVisible(bool(warnings))
        self._rebuild()

    def visible_clusters(self) -> list[DiffCluster]:
        """The findings that pass the filters and are not ignored (unless shown)."""
        category = self.category_combo.currentData()
        show_ignored = self.show_ignored_check.isChecked()
        show_uncertain = self.show_uncertain_check.isChecked()

        return [
            cluster
            for cluster in self._clusters
            if (show_ignored or not cluster.is_ignored)
            and (show_uncertain or not cluster.uncertain)
            and (category == _ALL or cluster.category == category)
        ]

    def marker_clusters(self) -> list[DiffCluster]:
        """What the viewer should draw: visible findings that are still active."""
        return [cluster for cluster in self.visible_clusters() if not cluster.is_ignored]

    def cluster(self, cluster_id: int) -> DiffCluster | None:
        return next((c for c in self._clusters if c.id == cluster_id), None)

    # -- merging --------------------------------------------------------------
    @property
    def can_undo_merge(self) -> bool:
        return bool(self._merges)

    def merge(self, cluster_ids: list[int]) -> DiffCluster | None:
        """Make the findings with these ids one finding (in the place of the first); None if fewer than two exist."""
        chosen = [cluster for cluster in self._clusters if cluster.id in set(cluster_ids)]
        if len(chosen) < 2:
            return None
        merged = merge_clusters(chosen)
        position = min(self._clusters.index(cluster) for cluster in chosen)
        self._clusters = [cluster for cluster in self._clusters if cluster not in chosen]
        self._clusters.insert(position, merged)
        self._merges.append((merged, chosen, position))
        self._selected_id = merged.id
        # Merging is the reviewer's decision about what counts as one difference in regulated artwork: leave a trace.
        logger.info(
            "findings merged",
            extra={"merged_ids": sorted(cluster.id for cluster in chosen), "into": merged.id, "category": merged.category},
        )
        self._rebuild()
        self.visibilityChanged.emit()
        return merged

    def undo_merge(self) -> bool:
        """Split the last merged finding back into the findings it was made of."""
        if not self._merges:
            return False
        merged, originals, position = self._merges.pop()
        if merged not in self._clusters:
            return False
        self._clusters.remove(merged)
        for offset, cluster in enumerate(originals):
            self._clusters.insert(position + offset, cluster)
        self._selected_id = originals[0].id
        logger.info("findings unmerged", extra={"merged_id": merged.id, "restored_ids": sorted(c.id for c in originals)})
        self._rebuild()
        self.visibilityChanged.emit()
        return True

    # -- interaction ----------------------------------------------------------
    def select(self, cluster_id: int | None) -> None:
        self._selected_id = cluster_id
        self._rebuild()
        if cluster_id is not None:
            self._scroll_to(cluster_id)

    def _on_card_selected(self, cluster_id: int) -> None:
        self.select(cluster_id)
        self.clusterSelected.emit(cluster_id)

    def _on_ignore_toggled(self, cluster_id: int) -> None:
        cluster = self.cluster(cluster_id)
        if cluster is None:
            return
        if cluster.is_ignored:
            cluster.restore()
        else:
            cluster.ignore()
        # Ignoring is a decision about regulated artwork, so leave a trace.
        logger.info(
            "finding %s",
            "ignored" if cluster.is_ignored else "restored",
            extra={
                "cluster_id": cluster.id,
                "category": cluster.category,
                "detail": cluster.detail,
            },
        )
        self._rebuild()
        self.visibilityChanged.emit()

    def _on_filters_changed(self, *_args) -> None:
        self._rebuild()
        self.visibilityChanged.emit()

    def _scroll_to(self, cluster_id: int) -> None:
        for index in range(self._cards_layout.count()):
            widget = self._cards_layout.itemAt(index).widget()
            if isinstance(widget, FindingCard) and widget.cluster_id == cluster_id:
                # The cards were just rebuilt and have no geometry until the
                # layout runs, so wait a turn of the event loop.
                QTimer.singleShot(0, lambda w=widget: self._scroll.ensureWidgetVisible(w))
                return

    # -- rendering ----------------------------------------------------------
    def _rebuild(self) -> None:
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # deleteLater() alone leaves the old card parented (and
                # briefly painted) until the next event-loop turn.
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

        visible = self.visible_clusters()
        for cluster in visible:
            card = FindingCard(cluster, selected=cluster.id == self._selected_id)
            card.selectRequested.connect(self._on_card_selected)
            card.ignoreToggled.connect(self._on_ignore_toggled)
            self._cards_layout.addWidget(card)

        if not visible:
            empty = QLabel(
                "No differences found."
                if not self._clusters
                else "No findings match the current filters."
            )
            empty.setProperty("secondary", "true")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._cards_layout.addWidget(empty)
        self._cards_layout.addStretch(1)

        self._update_header()

    def _update_header(self) -> None:
        active = [c for c in self._clusters if not c.is_ignored and (self.show_uncertain_check.isChecked() or not c.uncertain)]
        ignored = sum(1 for c in self._clusters if c.is_ignored)
        hidden = sum(1 for c in self._clusters if c.uncertain and not c.is_ignored and not self.show_uncertain_check.isChecked())
        self.title_label.setText(f"Findings ({len(active)})")

        # One chip per kind that is present, in that kind's colour: the same colour as its boxes in the two files.
        counts = [
            f'<span style="color:{category_colour(key)}">■</span> '
            f"{sum(1 for c in active if c.category == key)} {CATEGORY_LABELS[key]}"
            for key in CATEGORY_ORDER
            if any(c.category == key for c in active)
        ]
        if hidden:
            counts.append(f"{hidden} possible misread(s) hidden")
        if ignored:
            counts.append(f"{ignored} ignored")
        self.summary_label.setText("  ·  ".join(counts) if self._clusters else "")
