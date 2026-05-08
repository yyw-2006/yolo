# YOLOv8 训练思路

## 实现思路/结论

YOLOv8 更适合训练“可被框出来的原子物体”，例如枪支、刀具、注射器、扑克牌、骰子、纳粹符号等。项目最终需要输出的是业务类别，例如暴恐、违禁毒品、纳粹符号、赌博、游行集会、不雅手势、正常，因此不能把 YOLO 检测类别直接一对一映射成业务类别。更合理的结构是：先用 YOLOv8 检测基础物体，再通过规则或分类模型把多个检测结果组合成业务判断。像 gun、swastika、middle_finger 属于强证据，可以单独触发风险类别；像 flag、crowd、banner 属于弱证据，单独出现不能直接判定为游行集会。

推荐的整体流程：

```text
图片输入
→ YOLOv8 原子物体检测
→ 检测结果聚合与规则判断
→ 输出业务类别、置信度、命中证据
```

不要使用这种直接映射：

```text
flag → 游行集会
```

应该使用组合判断：

```text
flag + crowd + banner/signboard → 可能是游行集会
flag + crowd + protest_sign + police/riot_shield → 高置信游行集会
单独 flag → 正常或仅记录为普通旗帜特征
```

## 基础检测类与业务类别映射

建议先训练或整理这些基础检测类：

```text
gun
knife
blood
syringe
pill
powder
swastika
nazi_symbol
poker_card
dice
chip
roulette
slot_machine
middle_finger
crowd
flag
banner
signboard
police
riot_shield
```

业务类别可以通过规则聚合：

```python
if has("gun") or has("knife") or has("blood") or has("police") or has("riot_shield"):
    category = "暴恐"

elif has("syringe") or has("powder"):
    category = "违禁毒品"

elif has("swastika") or has("nazi_symbol"):
    category = "纳粹符号"

elif has("poker_card") or has("dice") or has("chip") or has("roulette") or has("slot_machine"):
    category = "赌博"

elif has("middle_finger"):
    category = "不雅手势"

elif (has("crowd") or face_count >= 8) and (has("flag") or has("banner") or has("signboard")):
    category = "游行集会"

else:
    category = "正常"
```

注意：规则只是第一版 demo 方案。涉政、非法宗教、游行集会这类语义更强的类别，后续更适合增加整图分类模型或 CLIP 类模型，结合 YOLO 检测结果一起判断。

## 可用 Roboflow 数据集候选

以下数据集来自 Roboflow Universe，需要下载后人工检查质量、统一标签名、删除明显错误标注，再合并训练。

| 业务类别 | 基础物体 | 候选数据集 | 备注 |
|---|---|---|---|
| 暴恐 | gun / knife / rifle / pistol | https://universe.roboflow.com/weopon-detection/weapon-detection-using-yolov8 | 约 671 张，包含 Knife、Handgun、Rifle、Shotgun、Sword 等 |
| 暴恐 | knife / gun / stick | https://universe.roboflow.com/violence-detection-rvh0k/knife-gun-stick-detection | 约 8.2k 张，适合作为武器检测候选 |
| 暴恐 | blood | https://universe.roboflow.com/bloodimage-byvud/blood-ijmoe | 约 216 张，数据量较小 |
| 暴恐 | blood | https://universe.roboflow.com/blood-u8rnv/b2-cm8xg-outea | 约 4.6k 张，但类别和质量需要重点清洗 |
| 违禁毒品 | syringe | https://universe.roboflow.com/search?q=class%3Asyringe | 搜索页中有 1848 张、282 张等多个候选 |
| 违禁毒品 | pill / tablet / capsule | https://universe.roboflow.com/seblful/pills-detection-s9ywn | 约 696 张，类别是 capsules、tablets |
| 违禁毒品 | pill | https://universe.roboflow.com/mohamed-attia-e2mor/pill-detection-llp4r | 约 451 到 1083 张，有多个版本 |
| 纳粹符号 | swastika / nazi symbols | https://universe.roboflow.com/zhiwei/nazi-symbols | 约 3125 张，包含多种纳粹相关符号 |
| 纳粹符号 | swastika | https://universe.roboflow.com/rm-yoq1a/swast | 约 2.8k 张，类别较多，需要筛选 |
| 赌博 | poker cards | https://universe.roboflow.com/roboflow-100/poker-cards-cxcvz | 约 1.3k 张，扑克牌检测 |
| 赌博 | dice | https://universe.roboflow.com/roboflow-gw7yv/dice | 约 359 张，公开骰子数据集 |
| 赌博 | dice | https://universe.roboflow.com/search?q=class%3Adice | 搜索页中还有更多骰子数据集候选 |
| 赌博 | poker chips | https://universe.roboflow.com/urop2023-ar-metaverse/poker-chips-detector | 筹码检测，可作为赌博物体证据 |
| 赌博 | roulette | https://universe.roboflow.com/yogajangkung/roulette-wheel | 约 455 张，轮盘相关检测 |
| 赌博 | casino | https://universe.roboflow.com/prolifics-pzsny/casino-yxutg | 约 200 张，赌场/老虎机局部元素 |

| 不雅手势 | middle finger | https://universe.roboflow.com/kuohuanchi-2vfnq/middle-finger-1vyzi | 约 55 张，数据量小，可辅助参考 |
| 游行集会 | riot / crowd | https://universe.roboflow.com/riot-detection/riot-det | 约 1.9k 张，含 Riot Threats、Pedestrian 等 |
| 游行集会 | crowd | https://universe.roboflow.com/yolo-analysis/crowd-counting-umiax | 约 2.2k 张，适合做人群证据 |
| 游行集会 | flag | https://universe.roboflow.com/search?q=class%3Aflag | 搜索页中有 flag、banner 等候选 |

## 使用方法/运行方式

第一阶段建议只做可控 demo：优先训练 gun、knife、syringe、pill、swastika、poker_card、dice、middle_finger 这些强证据类别。下载数据集时优先选择 Object Detection 类型，并导出 YOLOv8 格式。合并数据集前需要统一标签名，例如 handgun、pistol、rifle 是否合并为 gun，需要根据业务要求先确定。训练完成后不要直接输出 YOLO 类别，而是输出业务类别、命中的基础物体、置信度和触发规则，方便后续调阈值和排查误报。

第二阶段再处理弱语义类别，例如游行集会、涉政、非法宗教。对于这些类别，YOLO 只负责提供 crowd、flag、banner、signboard、police、riot_shield 等证据，最终判断建议交给规则组合或整图分类模型。这样可以避免“普通国旗被直接判成游行集会”这类误报。
