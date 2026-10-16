"""Embedded setup must localize without installing an app translator."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

from anylabeling.views.labeling.utils.auto_labeling_i18n import (
    auto_labeling_text_v1,
)
from anylabeling.views.labeling.widgets.auto_labeling_run_dialog import (
    ContinuousRunSetupDialog, FastRunProgressDialog,
)


class ContinuousSetupLocalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.parent = QtWidgets.QWidget()
        self.parent._config = {"language": "en_US", "review_language": "zh_CN"}
        self.dialog = ContinuousRunSetupDialog(
            {"workset_total": 218, "existing_annotation": 34,
             "host_prepare_failed": 0},
            {"model_name": "0.75_first_ln_89d5f212",
             "confidence_threshold": 0.255, "iou_threshold": 0.45},
            current_anchor_available=True, parent=self.parent,
        )

    def tearDown(self):
        self.dialog.close()
        self.parent.close()

    def test_embedded_chinese_text_and_readonly_parameters(self):
        dialog = self.dialog
        self.assertEqual(dialog.windowTitle(), "批量自动标注")
        self.assertEqual(dialog.range_combo.currentText(), "从当前图片到最后一张")
        self.assertEqual(dialog.mode_combo.currentText(), "逐张预览结果")
        self.assertEqual(dialog.delay_spin.suffix(), " 秒")
        self.assertEqual(dialog.cancel_button.text(), "取消")
        self.assertEqual(dialog.start_button.text(), "开始标注并预览")
        labels = [label.text() for label in dialog.findChildren(QtWidgets.QLabel)]
        for text in ("最低可信度", "25.5%", "重叠框去重阈值", "45%",
                     "列表图片总数", "218", "无法准备的图片"):
            self.assertIn(text, labels)
        self.assertFalse(any("conf=" in text for text in labels))
        self.assertTrue(dialog.note_label.isHidden())
        self.assertEqual(dialog.selected_values(), {
            "delay_seconds": 2.0, "range": "CURRENT_TO_END",
            "filter": "ALL", "write_policy": "SKIP_EXISTING",
        })

    def test_modes_keep_preview_delay_without_changing_other_options(self):
        dialog = self.dialog
        dialog.delay_spin.setValue(3.5)
        dialog.mode_combo.setCurrentIndex(1)
        self.assertEqual(dialog.selected_values()["delay_seconds"], 0.0)
        self.assertFalse(dialog.delay_spin.isEnabled())
        self.assertEqual(dialog.start_button.text(), "开始后台标注")
        dialog.mode_combo.setCurrentIndex(0)
        self.assertEqual(dialog.delay_spin.value(), 3.5)
        self.assertTrue(dialog.delay_spin.isEnabled())
        self.assertEqual(dialog.selected_values()["write_policy"], "SKIP_EXISTING")

    def test_progress_language_and_status_follow_embedded_parent(self):
        progress = FastRunProgressDialog(self.parent)
        try:
            progress.set_phase("INFERENCING")
            self.assertEqual(progress.state_label.text(), "正在识别目标")
            progress.update_progress({
                "total": 218, "processed": 12, "delay_seconds": 2,
                "processing_status": "RUNNING", "completed_with_errors": False,
            })
            self.assertEqual(progress.metric_labels["processing_status"].text(), "运行中")
            self.assertEqual(progress.metric_labels["completed_with_errors"].text(), "否")
            self.assertEqual(progress.configured_delay_label.text(), "2.0 秒")
        finally:
            progress.close()

    def test_fast_entry_uses_default_preview_delay_when_switched(self):
        dialog = ContinuousRunSetupDialog(
            {}, "YOLO", current_anchor_available=True,
            initial_values={"delay_seconds": 0}, parent=self.parent,
        )
        try:
            self.assertEqual(dialog.mode_combo.currentData(), "FAST")
            dialog.mode_combo.setCurrentIndex(0)
            self.assertEqual(dialog.delay_spin.value(), 2.0)
        finally:
            dialog.close()

    def test_explicit_language_does_not_change_global_or_other_windows(self):
        before = auto_labeling_text_v1("continuous_title")
        english_parent = QtWidgets.QWidget()
        english_parent._config = {"language": "en_US"}
        english = ContinuousRunSetupDialog(
            {}, "YOLO", current_anchor_available=False,
            parent=english_parent,
        )
        try:
            self.assertEqual(english.windowTitle(), "Continuous auto-labeling")
            self.assertEqual(english.range_combo.currentData(), "ALL_IMAGES")
            self.assertFalse(english.range_combo.model().item(1).isEnabled())
            self.assertEqual(self.dialog.windowTitle(), "批量自动标注")
            self.assertEqual(auto_labeling_text_v1("continuous_title"), before)
        finally:
            english.close()
            english_parent.close()
