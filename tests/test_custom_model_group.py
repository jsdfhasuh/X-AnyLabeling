import copy
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets

from anylabeling.views.labeling.widgets.auto_labeling import auto_labeling
from anylabeling.views.labeling.widgets.searchable_model_dropdown import (
    ProviderSection,
    SearchableModelDropdownPopup,
)


class CustomModelGroupTests(unittest.TestCase):
    def setUp(self):
        self.models = [
            {
                "name": "project_model",
                "display_name": "83_kaggle_20260822",
                "provider": "custom",
                "config_file": "project_model.yaml",
                "is_custom_model": False,
            },
            {
                "name": "_custom_imported",
                "display_name": "Imported Model",
                "provider": "Ultralytics",
                "config_file": "imported.yaml",
                "is_custom_model": True,
            },
            {
                "name": "remote_server",
                "display_name": "Remote-Server",
                "provider": "CVHub",
                "config_file": ":/remote_server.yaml",
            },
        ]
        self.host = SimpleNamespace(
            model_manager=mock.Mock(),
            _sort_model_data=lambda data: (
                auto_labeling.AutoLabelingWidget._sort_model_data(None, data)
            ),
        )
        self.host.model_manager.get_model_configs.return_value = self.models

    def build(self, saved=None):
        with mock.patch.object(
            auto_labeling,
            "load_json",
            return_value={"models_data": copy.deepcopy(saved or {})},
        ):
            return auto_labeling.AutoLabelingWidget.init_model_data(self.host)

    def test_custom_aliases_merge_into_first_group(self):
        for provider in ("custom", "Custom", "CUSTOM"):
            with self.subTest(provider=provider):
                self.models[0]["provider"] = provider
                data = self.build()
                self.assertEqual(list(data), ["Custom", "CVHub"])
                self.assertEqual(
                    list(data["Custom"]),
                    ["load_custom_model", "project_model", "_custom_imported"],
                )

    def test_legacy_lowercase_favorite_survives_merge_and_reload(self):
        data = self.build(
            {"custom": {"project_model": {"favorite": True}}}
        )
        self.assertTrue(data["Custom"]["project_model"]["favorite"])
        reloaded = self.build(data)
        self.assertEqual(reloaded, data)
        self.assertNotIn("_custom_project_model", self.host.model_info)

    def test_stale_cached_models_are_not_reintroduced(self):
        data = self.build(
            {
                "custom": {
                    "project_model": {"favorite": True},
                    "removed_model": {"favorite": True},
                }
            }
        )
        self.assertNotIn("removed_model", data["Custom"])

    def test_imported_model_keeps_saved_config_path_on_reload(self):
        config_path = str(Path(__file__).resolve())
        data = self.build(
            {
                "Custom": {
                    "_custom_imported": {
                        "favorite": True,
                        "display_name": "Imported Model",
                        "config_path": config_path,
                    }
                }
            }
        )
        self.assertEqual(
            data["Custom"]["_custom_imported"]["config_path"], config_path
        )
        self.assertEqual(self.build(data), data)

    def test_saved_file_model_is_restored_after_catalog_entry(self):
        config_path = str(Path(__file__).resolve())
        data = self.build(
            {
                "Custom": {
                    "project_model": {"favorite": True},
                    "saved_model": {
                        "favorite": True,
                        "display_name": "Saved Model",
                        "config_path": config_path,
                    },
                }
            }
        )
        self.assertIn("_custom_saved_model", data["Custom"])
        self.assertTrue(
            self.host.model_info["_custom_saved_model"]["is_custom_model"]
        )

    def test_group_merge_preserves_both_loading_routes(self):
        self.build()
        self.host._configure_remote_server_model = lambda name: True
        self.host.clear_auto_labeling_action_requested = mock.Mock()
        self.host.model_selection_button = mock.Mock()
        self.host.hide_labeling_widgets = mock.Mock()
        self.host.new_model_selected = mock.Mock()

        auto_labeling.AutoLabelingWidget.on_model_selected(
            self.host, "Custom", "project_model"
        )
        self.host.new_model_selected.emit.assert_called_once_with(
            "project_model.yaml"
        )
        self.host.model_manager.load_custom_model.assert_not_called()

        auto_labeling.AutoLabelingWidget.on_model_selected(
            self.host, "Custom", "_custom_imported"
        )
        self.host.model_manager.load_custom_model.assert_called_once_with(
            "imported.yaml"
        )
        self.host.new_model_selected.emit.assert_called_once()

    def test_custom_models_render_in_first_section_and_can_be_selected(self):
        app = (
            QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        )
        data = self.build(
            {"custom": {"project_model": {"favorite": True}}}
        )
        popup = SearchableModelDropdownPopup(data)
        try:
            popup.show()
            app.processEvents()
            sections = [
                popup.container_layout.itemAt(i).widget()
                for i in range(popup.container_layout.count())
                if isinstance(
                    popup.container_layout.itemAt(i).widget(), ProviderSection
                )
            ]
            headings = [
                section.layout().itemAt(0).layout().itemAt(1).widget().text()
                for section in sections
            ]
            self.assertEqual(headings, ["Custom", "Favorites", "CVHub"])
            scroll = popup.findChild(QtWidgets.QScrollArea)
            viewport = scroll.viewport()
            for name in ("load_custom_model", "project_model"):
                item = popup.model_items[name]
                rect = QtCore.QRect(
                    item.mapTo(viewport, QtCore.QPoint()), item.size()
                )
                self.assertTrue(viewport.rect().contains(rect))
            screenshot = os.environ.get("CUSTOM_MODEL_SCREENSHOT")
            if screenshot:
                self.assertTrue(popup.grab().save(screenshot))
            selected = QtTest.QSignalSpy(popup.modelSelected)
            with mock.patch.object(popup, "save_models_data"):
                QtTest.QTest.mouseClick(
                    popup.model_items["project_model"], QtCore.Qt.LeftButton
                )
            self.assertEqual(list(selected), [["Custom", "project_model"]])
        finally:
            popup.close()
            popup.deleteLater()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
