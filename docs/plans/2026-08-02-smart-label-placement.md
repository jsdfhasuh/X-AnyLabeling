# 智能标签外置与边界避让改动计划

- 日期：2026-08-02
- 仓库：`jsdfhasuh/X-AnyLabeling`
- 实施分支：`codex/smart-label-placement`
- 基线分支：`codex/project-pose-annotation-support`
- 状态：可开始实施

## 1. 背景与问题

当前 `Canvas.paintEvent()` 在绘制 `rectangle`、`polygon`、`rotation` 标签时，直接以图形包围框的左上角作为标签背景起点。标签背景和文字因此覆盖图形左上角边线、选中顶点手柄以及框内小目标，用户很难确认标注框是否准确贴合目标。

本问题属于画布显示层问题，不属于标注数据、模型推理或训练格式问题。

## 2. 目标

1. `rectangle`、`polygon`、`rotation` 的标签默认绘制在图形外部，不再覆盖框内目标。
2. 标签必须避开自己的边框和选中顶点手柄。
3. 标签靠近图像四边时自动换位，不能被画布裁掉。
4. 图片缩放时，标签字号、内边距以及标签与图形的间隔保持稳定的屏幕视觉尺寸。
5. 显示描述或属性时，标签位置不能与描述、属性区域直接重叠。
6. 修正当前标签绘制中的可见性判断和遗留循环变量问题。
7. 不改变已有标注文件、YOLO 导入导出、训练逻辑和模型行为。

## 3. 非目标

本轮明确不做以下内容：

- 不修改 `Shape.to_dict()`、`Shape.load_from_dict()` 或任何标注 JSON 字段。
- 不保存标签显示位置，标签位置始终由画布实时计算。
- 不改变 `show_masks`、遮罩透明度或 `Ctrl+M` 行为。
- 不做所有标签之间的全局避碰和自动排版。
- 不改变圆、点、直线、折线的既有标签布局。
- 不整体同步或合并上游 PyQt6 代码，只参考其固定屏幕字号的绘制思路。
- 不在本次子仓库任务中更新父仓库 `emo-vision-train` 的子模块指针。

## 4. 当前代码结论

主要修改点为：

```text
anylabeling/views/labeling/widgets/canvas.py
```

当前标签绘制逻辑位于 `Canvas.paintEvent()` 的 `# Draw labels` 区段。对于 `rectangle`、`polygon`、`rotation`，标签矩形起点使用 `bbox.x()`、`bbox.y()`，导致标签位于包围框内部左上角。

`Shape.paint()` 只负责图形边线、填充和顶点，不应承载标签布局逻辑，因此本轮不修改：

```text
anylabeling/views/labeling/shape.py
```

现有代码还包含以下局部问题，应随本轮一起修正：

1. 同一标签生成循环重复检查 `shape.visible`。
2. 绘制文字的第二个循环没有携带 `shape`，却访问前一循环遗留的 `shape.visible`。
3. 标签绘制只检查 `shape.visible`，未统一使用 `self.is_visible(shape)`，可能导致过滤隐藏后仍残留标签。

## 5. 设计方案

### 5.1 使用屏幕坐标绘制标签

标签布局和绘制必须从图像坐标切换为屏幕坐标：

1. 在进入标签绘制前保存当前 `QPainter` 变换。
2. 使用当前变换把图形包围框和图像范围映射为屏幕坐标。
3. `p.save()` 后执行 `p.resetTransform()`。
4. 使用固定 UI 字号、固定屏幕像素内边距和间隔绘制标签。
5. 使用 `try/finally` 或等价结构保证 `p.restore()` 必定执行。

建议结构：

```python
label_transform = painter.transform()
painter.save()
try:
    painter.resetTransform()
    ...
finally:
    painter.restore()
```

这样图片在 50%、100%、300% 等缩放比例下，标签不会跟随图片无限放大或缩小。

### 5.2 标签候选位置

仅对 `rectangle`、`polygon`、`rotation` 使用智能外置布局。

默认候选顺序：

1. 左上方外侧。
2. 左下方外侧。
3. 右上方外侧。
4. 左上侧外侧。
5. 框内左上角作为最终兜底。

候选位置之间保留固定屏幕间隔，建议初始值为 3 个逻辑像素。

当图形存在描述且 `show_texts=True` 时，上方候选应降级到后面；当图形存在属性且 `show_attributes=True` 时，下方候选应降级到后面。不要为了避开描述或属性而修改它们当前的绘制逻辑。

### 5.3 图形保护区域

候选标签不能与图形保护区域相交。保护区域由图形屏幕包围框向外扩展得到，扩展量至少覆盖：

- 标签与边框的固定间隔；
- 选中顶点手柄的一半屏幕尺寸；
- 1 个像素的取整误差余量。

即使图形未选中，也使用同一保护区域，避免选中后标签突然跳动。

### 5.4 图像边界约束

标签必须完整位于映射后的图像屏幕矩形内，而不是仅限制在整个窗口内。候选位置不满足完整包含条件时，应尝试下一个候选位置。

不能通过把上方标签简单向下夹紧的方式重新压回图形内部。只有所有外部候选都失败时，才允许使用框内兜底位置。

### 5.5 超长标签

标签原始内容不得修改。绘制层应根据图像当前可用屏幕宽度使用 `QFontMetrics.elidedText()` 生成显示文本。

要求：

- 真实 `shape.label`、`group_id` 和 `score` 不变。
- 标签显示文本最长不得超过图像屏幕宽度减去两侧安全边距。
- 窄画布下允许显示省略号。

### 5.6 颜色与显示开关

保持现有标签视觉语义：

- 背景继续使用 `shape.line_color`。
- 文字继续使用黑色。
- `show_labels`、`show_scores` 行为保持不变。
- 自动标注内部标签 `AUTOLABEL_OBJECT`、`AUTOLABEL_ADD`、`AUTOLABEL_REMOVE` 继续跳过。
- 图形同时满足 `shape.visible` 和 `self.is_visible(shape)` 才绘制标签。

本轮不重新设计标签颜色、对比度或透明度。

### 5.7 不做全局标签碰撞

首版只保证标签不遮挡自己的图形，不保证不同图形的标签彼此不重叠。

原因：全局避碰会增加每帧绘制复杂度，并可能在密集标注、拖动和缩放时造成标签持续跳动。待本轮稳定后再单独评估。

## 6. 代码改动清单

### 6.1 `canvas.py`

在模块顶部增加少量标签显示常量，例如：

```python
LABEL_FONT_POINT_SIZE = 8.0
LABEL_PADDING_X_PX = 4
LABEL_PADDING_Y_PX = 2
LABEL_GAP_PX = 3
LABEL_EDGE_MARGIN_PX = 1
```

将 `paintEvent()` 中完整的 `# Draw labels` 区段抽离为私有方法，避免继续增加主绘制函数复杂度。建议方法职责如下，具体命名可根据周边风格微调：

```python
def _build_shape_label_text(self, shape):
    """组合 group_id、label 和可选 score。"""


def _shape_label_screen_rect(self, shape, transform):
    """把图形包围框映射为屏幕坐标。"""


def _label_candidate_rects(
    self,
    shape_rect,
    label_size,
    prefer_top,
    prefer_bottom,
):
    """按稳定顺序生成外置候选位置。"""


def _place_shape_label(
    self,
    shape,
    shape_rect,
    label_size,
    image_rect,
):
    """选择第一个完整可见且不与保护区域相交的位置。"""


def _draw_shape_labels(self, painter):
    """统一完成标签文本、布局、背景和文字绘制。"""
```

实现要求：

- 使用 `QRectF` 完成候选计算和相交判断，最终绘制前再做整数取整。
- 标签背景和文字应在同一个标签循环中连续绘制，避免依赖前一循环的 `shape` 变量。
- 对没有点、包围框无效或映射后尺寸无效的图形安全跳过。
- 不修改图形对象，不产生 dirty 状态，不进入撤销栈。
- 继续支持拖动、缩放、旋转时实时更新。

### 6.2 `label_widget.py`

更新 View 菜单提示，避免继续描述为“显示在图形内部”。建议：

```text
Show labels near shapes
Show scores with labels
```

只调整提示文字，不增加新的菜单项或配置开关。

### 6.3 翻译文件

同步更新：

```text
anylabeling/resources/translations/en_US.ts
anylabeling/resources/translations/zh_CN.ts
```

中文建议：

```text
在图形附近显示标签
随标签显示置信度
```

如项目当前发布流程需要 `.qm` 生成步骤，按现有资源流程执行；不要手工修改自动生成的大型 Python 资源文件。

### 6.4 测试

新增：

```text
tests/test_canvas/test_label_placement.py
```

测试使用 PyQt5 离屏模式：

```python
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
```

优先测试抽离后的布局方法，另增加少量真实 `Canvas.grab()` 渲染测试。

### 6.5 变更记录

在 `CHANGELOG.md` 当前版本顶部增加一条改进说明：

```text
- Place shape labels outside bounding boxes with boundary-aware fallback to improve small-object inspection.
```

不要调整版本号。

## 7. 测试矩阵

至少覆盖以下情况：

1. 普通矩形：标签位于左上方外侧，且不与图形相交。
2. 顶部边界：上方空间不足时自动放到底部。
3. 底部边界：下方空间不足时自动放到右侧或左侧。
4. 左上角、右上角、左下角、右下角：标签始终完整处于图像范围内。
5. 小尺寸框：标签不覆盖框内目标和四个边角。
6. 多边形：使用包围框外置标签。
7. 旋转框：使用旋转图形包围框外置标签。
8. 描述开启：存在描述时不优先抢占上方区域。
9. 属性开启：存在属性时不优先抢占下方区域。
10. 超长标签：显示省略号，真实标签内容不变。
11. 分数开关：`show_scores` 只改变文字内容，不破坏位置约束。
12. 图形过滤隐藏：`self.is_visible(shape) == False` 时不绘制标签。
13. 图形自身隐藏：`shape.visible == False` 时不绘制标签。
14. 缩放稳定：50%、100%、300% 下标签视觉字号和间隔基本一致。
15. 其他形状：circle、point、line、linestrip 保持现有布局。
16. 遮罩开关：`show_masks` 开关不改变标签位置。
17. 数据兼容：保存前后不新增标签位置字段，已有 JSON 内容结构不变。

## 8. 人工验收场景

使用包含 `cal_hole1`、`cal_hole2`、`target_hole` 等小孔目标的样例图进行人工验收。

必须满足：

- 标签背景不覆盖孔洞。
- 框的四条边完整可见。
- 选中时四个顶点手柄完整可见。
- 能明确判断目标是否完全落在标注框内。
- 拖动或缩放框时标签稳定跟随，不出现明显闪烁和来回跳位。
- 靠近图像边缘时标签不被裁掉。
- `Ctrl+L`、`Ctrl+M`、分数显示及描述/属性显示仍正常。

## 9. 验证命令

在父工程本地目录 `D:\training_platform\X-AnyLabeling` 中，优先使用指定 Conda 解释器：

```bat
set QT_QPA_PLATFORM=offscreen

C:\Users\jsdfhasuh\.conda\envs\training_platform\python.exe -m pytest tests/test_canvas/test_label_placement.py -q

C:\Users\jsdfhasuh\.conda\envs\training_platform\python.exe -m pytest tests/test_utils/test_general.py -q

C:\Users\jsdfhasuh\.conda\envs\training_platform\python.exe -m flake8 anylabeling/views/labeling/widgets/canvas.py anylabeling/views/labeling/label_widget.py tests/test_canvas/test_label_placement.py

C:\Users\jsdfhasuh\.conda\envs\training_platform\python.exe -m black --check anylabeling/views/labeling/widgets/canvas.py anylabeling/views/labeling/label_widget.py tests/test_canvas/test_label_placement.py
```

先运行新增单测，再运行相关既有测试。不要在尚未确认依赖完整前把全量模型测试失败误判为本次画布改动失败。

## 10. 风险与防护

### Painter 状态泄漏

风险：`resetTransform()` 后异常退出可能影响后续十字线、属性和 Compare View 绘制。

防护：必须使用 `save/restore`，并保证任何分支都能恢复 Painter 状态。

### 坐标取整造成 1 像素相交

风险：候选计算为浮点，最终整数绘制后可能重新压到框线上。

防护：相交判断前扩大保护区域，并保留 1 像素余量。

### 密集标注中的标签重叠

风险：不同图形的外置标签仍可能彼此覆盖。

处理：本轮接受该限制，不引入全局避碰；验收只要求不覆盖自己的目标框。

### 与姿态标注分支冲突

风险：当前基线分支已经修改 `canvas.py` 和 `label_widget.py`。

防护：本分支直接从 `codex/project-pose-annotation-support` 创建，实施时保留姿态骨架绘制、姿态配置和相关测试，不回退这些改动。

## 11. 完成定义

只有同时满足以下条件才算完成：

- 计划中的布局规则已落地。
- 新增测试覆盖主要边界场景并通过。
- 相关既有测试通过。
- Black 和 flake8 对改动文件通过。
- 小孔截图场景人工验收通过。
- 未修改标注数据结构、训练协议或模型逻辑。
- 未回退姿态标注分支已有功能。
- 提交记录按功能拆分清晰，最终汇报包含改动文件、测试结果、已知限制和提交 SHA。

## 12. 父仓库后续集成

子仓库实现和审查完成后，再在 `jsdfhasuh/emo-vision-train` 中单独更新 `X-AnyLabeling` 子模块指针，并运行完整桌面应用进行集成验证。本任务实施阶段不要提前修改父仓库。