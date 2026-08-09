import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml
from PyQt5 import QtCore

from anylabeling.views.labeling.utils import batch
from anylabeling.views.labeling.utils.auto_labeling_i18n import (
    AUTO_LABELING_TEXT_V1,
    TRANSLATION_CONTEXT_V1,
    auto_labeling_text_v1,
)
from anylabeling.views.labeling.utils.auto_labeling_sequence import (
    LEGACY_BATCH_ROUTE_V1,
    UNIFIED_SEQUENCE_ROUTE_V1,
    resolve_sequence_capability_decision_v1,
)
from anylabeling.views.labeling.widgets.auto_labeling.auto_labeling import (
    AutoLabelingWidget,
)


ROOT = Path(__file__).resolve().parents[1]


class Phase8CapabilityMatrixTests(unittest.TestCase):
    def test_detect_and_pose_are_the_only_verified_sequence_families(self):
        cases = {
            "yolov5": "YOLO_DETECT",
            "yolov8": "YOLO_DETECT",
            "yolo11": "YOLO_DETECT",
            "yolo26": "YOLO_DETECT",
            "yolov8_pose": "YOLO_POSE",
            "yolo11_pose": "YOLO_POSE",
        }
        for model_type, family in cases.items():
            with self.subTest(model_type=model_type):
                decision = resolve_sequence_capability_decision_v1(
                    {"type": model_type, "model": object()}
                )
                self.assertEqual(decision.model_family, family)
                self.assertEqual(
                    decision.route,
                    UNIFIED_SEQUENCE_ROUTE_V1,
                )
                self.assertTrue(decision.capabilities.supports_fast_sequence)
                self.assertTrue(
                    decision.capabilities.supports_visible_sequence
                )
                self.assertTrue(decision.creates_unified_audit_queue)
                self.assertIsNone(decision.reason_code)

    def test_unverified_or_stateful_families_remain_legacy(self):
        cases = {
            "yolov8_seg": (
                "SEGMENTATION",
                "capability_not_validated",
            ),
            "yolo11_obb": ("OBB", "capability_not_validated"),
            "yolov8_det_track": (
                "TRACKER",
                "stateful_across_images",
            ),
            "segment_anything_2_video": (
                "VIDEO",
                "stateful_across_images",
            ),
            "segment_anything": (
                "INTERACTIVE_SAM",
                "interactive_input_required",
            ),
            "remote_server": (
                "REMOTE_MULTI_IMAGE",
                "remote_multi_image_unverified",
            ),
        }
        for model_type, (family, reason) in cases.items():
            with self.subTest(model_type=model_type):
                decision = resolve_sequence_capability_decision_v1(
                    {"type": model_type, "model": object()}
                )
                self.assertEqual(decision.model_family, family)
                self.assertEqual(decision.route, LEGACY_BATCH_ROUTE_V1)
                self.assertEqual(decision.reason_code, reason)
                self.assertFalse(decision.capabilities.supports_fast_sequence)
                self.assertFalse(
                    decision.capabilities.supports_visible_sequence
                )
                self.assertFalse(decision.creates_unified_audit_queue)

    @staticmethod
    def _widget(model_type, opener):
        manager = SimpleNamespace(
            loaded_model_config={"type": model_type, "model": object()},
            new_model_status=SimpleNamespace(emit=mock.Mock()),
        )
        context = SimpleNamespace(activate_sequence_run=mock.Mock())
        auto_widget = SimpleNamespace(
            model_manager=manager,
            open_continuous_auto_labeling=opener,
            auto_labeling_host_context=context,
        )
        return SimpleNamespace(
            auto_labeling_widget=auto_widget,
            image_list=["image.jpg"],
            tr=lambda value: value,
        )

    def test_legacy_route_says_no_audit_queue_and_creates_no_run(self):
        for model_type in (
            "yolov8_seg",
            "yolo11_obb",
            "yolov8_det_track",
            "segment_anything_2_video",
            "segment_anything",
            "remote_server",
        ):
            with self.subTest(model_type=model_type):
                opener = mock.Mock()
                widget = self._widget(model_type, opener)
                with mock.patch.object(
                    batch,
                    "run_all_images_legacy",
                    return_value="legacy",
                ) as legacy:
                    self.assertEqual(batch.run_all_images(widget), "legacy")
                legacy.assert_called_once_with(widget)
                opener.assert_not_called()
                widget.auto_labeling_widget.auto_labeling_host_context.activate_sequence_run.assert_not_called()
                notice = widget.auto_labeling_widget.model_manager.new_model_status.emit.call_args.args[
                    0
                ]
                self.assertIn("does not create a unified audit queue", notice)

    def test_fast_failure_never_falls_back_to_legacy(self):
        for result in (False, RuntimeError("fast failed")):
            with self.subTest(result=repr(result)):
                opener = mock.Mock()
                if isinstance(result, Exception):
                    opener.side_effect = result
                else:
                    opener.return_value = result
                widget = self._widget("yolov8", opener)
                with mock.patch.object(
                    batch,
                    "run_all_images_legacy",
                ) as legacy:
                    if isinstance(result, Exception):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "fast failed",
                        ):
                            batch.run_all_images(widget)
                    else:
                        self.assertFalse(batch.run_all_images(widget))
                opener.assert_called_once_with()
                legacy.assert_not_called()

    def test_bound_resume_run_keeps_continuous_entry_available(self):
        button = mock.Mock()
        widget = SimpleNamespace(
            model_manager=SimpleNamespace(
                loaded_model_config={"type": "yolov8", "model": object()}
            ),
            auto_labeling_host_context=SimpleNamespace(
                active_run_id="run-resume",
                images_ready=True,
            ),
            _fast_run_session=None,
            button_continuous_run=button,
        )

        AutoLabelingWidget.refresh_continuous_run_availability(widget)

        button.setEnabled.assert_called_once_with(True)


class Phase8ConfigTranslationAndDocsTests(unittest.TestCase):
    def test_default_continuous_settings_are_frozen(self):
        config_path = (
            ROOT / "anylabeling" / "configs" / "xanylabeling_config.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            config["continuous_auto_labeling"],
            {
                "delay_seconds": 2.0,
                "range": "CURRENT_TO_END",
                "filter": "ALL",
                "write_policy": "INHERIT_MODEL_POLICY",
            },
        )

    def test_phase8_catalogs_cover_every_registered_source(self):
        translations = ROOT / "anylabeling" / "resources" / "translations"
        for language in ("en_US", "zh_CN"):
            with self.subTest(language=language):
                root = ET.parse(translations / f"{language}.ts").getroot()
                contexts = {
                    context.findtext("name"): context
                    for context in root.findall("context")
                }
                self.assertIn(TRANSLATION_CONTEXT_V1, contexts)
                messages = {}
                for message in contexts[TRANSLATION_CONTEXT_V1].findall(
                    "message"
                ):
                    translation = message.find("translation")
                    messages[message.findtext("source")] = translation
                self.assertEqual(
                    set(AUTO_LABELING_TEXT_V1.values()),
                    set(messages),
                )
                for source, translation in messages.items():
                    with self.subTest(language=language, source=source):
                        self.assertIsNotNone(translation)
                        self.assertNotEqual(
                            translation.attrib.get("type"),
                            "unfinished",
                        )
                        self.assertTrue((translation.text or "").strip())

    def test_compiled_qm_catalogs_translate_phase8_statuses(self):
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtCore.QCoreApplication([])
        translations = ROOT / "anylabeling" / "resources" / "translations"
        expected = {
            "en_US": "Staged approved",
            "zh_CN": "Session 已通过",
        }
        for language, translated in expected.items():
            with self.subTest(language=language):
                translator = QtCore.QTranslator()
                self.assertTrue(
                    translator.load(str(translations / f"{language}.qm"))
                )
                app.installTranslator(translator)
                try:
                    self.assertEqual(
                        auto_labeling_text_v1("status_staged_approved"),
                        translated,
                    )
                finally:
                    app.removeTranslator(translator)

    def test_user_guides_and_changelog_document_safety_boundaries(self):
        for path in (
            ROOT / "docs" / "en" / "user_guide.md",
            ROOT / "docs" / "zh_cn" / "user_guide.md",
        ):
            content = path.read_text(encoding="utf-8")
            for marker in (
                "Fast",
                "Visible",
                "Legacy Batch",
                "staged_approved",
                "approved",
                "Dataset",
                "waiver",
            ):
                with self.subTest(path=path.name, marker=marker):
                    self.assertIn(marker, content)
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn("Unified continuous auto-labeling", changelog)
        self.assertIn("does not create a unified audit queue", changelog)

    def test_compiled_translation_catalogs_are_packaged(self):
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        self.assertIn(
            "recursive-include anylabeling/resources/translations *.qm",
            manifest.splitlines(),
        )


if __name__ == "__main__":
    unittest.main()
