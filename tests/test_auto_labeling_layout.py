import os
from pathlib import Path
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

import anylabeling.config as anylabeling_config
from anylabeling.config import get_config
from anylabeling.views.labeling.label_widget import LabelingWidget


ROOT = Path(__file__).resolve().parents[1]


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _widget():
    app = _app()
    main_window = QtWidgets.QMainWindow()
    parent = SimpleNamespace(parent=main_window)
    previous_config_file = anylabeling_config.current_config_file
    config_yaml = (
        ROOT / "anylabeling" / "configs" / "xanylabeling_config.yaml"
    ).read_text(encoding="utf-8")
    anylabeling_config.current_config_file = config_yaml
    widget = LabelingWidget(parent=parent, config=get_config(config_yaml))
    widget._test_previous_config_file = previous_config_file
    widget.show()
    app.processEvents()
    return app, main_window, widget


class AutoLabelingLayoutTests(unittest.TestCase):
    def test_auto_labeling_widget_is_above_canvas_in_central_layout(self):
        _app_instance, main_window, widget = _widget()
        try:
            central_layout = widget.layout().itemAt(1).layout()
            widget_indexes = {
                central_layout.itemAt(index).widget(): index
                for index in range(central_layout.count())
                if central_layout.itemAt(index).widget() is not None
            }
            self.assertLess(
                widget_indexes[widget.label_instruction],
                widget_indexes[widget.auto_labeling_widget],
            )
            self.assertLess(
                widget_indexes[widget.auto_labeling_widget],
                widget_indexes[widget._central_widget],
            )
        finally:
            widget.close()
            main_window.close()
            anylabeling_config.current_config_file = (
                widget._test_previous_config_file
            )

    def test_right_sidebar_has_labels_and_review_only(self):
        _app_instance, main_window, widget = _widget()
        try:
            tab_texts = [
                widget.right_sidebar_tabs.tabText(index)
                for index in range(widget.right_sidebar_tabs.count())
            ]
            self.assertEqual(widget.right_sidebar_tabs.count(), 2)
            self.assertIn("Labels", tab_texts)
            self.assertTrue(any("Review" in text for text in tab_texts))
            self.assertNotIn("Auto labeling", tab_texts)
            self.assertFalse(hasattr(widget, "auto_labeling_sidebar_page"))
        finally:
            widget.close()
            main_window.close()
            anylabeling_config.current_config_file = (
                widget._test_previous_config_file
            )

    def test_ai_action_toggles_top_bar_and_run_action(self):
        _app_instance, main_window, widget = _widget()
        try:
            self.assertTrue(widget.auto_labeling_widget.isHidden())
            self.assertFalse(widget.actions.run_all_images.isEnabled())

            widget.toggle_auto_labeling_widget()
            self.assertTrue(widget.auto_labeling_widget.isVisible())
            self.assertTrue(widget.actions.run_all_images.isEnabled())

            widget.toggle_auto_labeling_widget()
            self.assertTrue(widget.auto_labeling_widget.isHidden())
            self.assertFalse(widget.actions.run_all_images.isEnabled())
        finally:
            widget.close()
            main_window.close()
            anylabeling_config.current_config_file = (
                widget._test_previous_config_file
            )

    def test_review_tab_switch_does_not_hide_top_bar_or_disable_run(self):
        _app_instance, main_window, widget = _widget()
        try:
            widget.update_thumbnail_display = lambda: None
            widget.toggle_auto_labeling_widget()
            self.assertTrue(widget.auto_labeling_widget.isVisible())
            self.assertTrue(widget.actions.run_all_images.isEnabled())

            widget.right_sidebar_tabs.setCurrentWidget(
                widget.audit_sidebar_page
            )
            _app().processEvents()
            self.assertTrue(widget.auto_labeling_widget.isVisible())
            self.assertTrue(widget.actions.run_all_images.isEnabled())

            widget.right_sidebar_tabs.setCurrentWidget(
                widget.labels_sidebar_page
            )
            _app().processEvents()
            self.assertTrue(widget.auto_labeling_widget.isVisible())
            self.assertTrue(widget.actions.run_all_images.isEnabled())
        finally:
            widget.close()
            main_window.close()
            anylabeling_config.current_config_file = (
                widget._test_previous_config_file
            )


if __name__ == "__main__":
    unittest.main()
