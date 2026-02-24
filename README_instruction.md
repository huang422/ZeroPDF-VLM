# VLM PDF Recognizer 操作手冊

本文件說明如何安裝、執行、維護此專案，以及如何新增模板與 ROI 欄位。

---

## 系統架構簡介

```
掃描文件 (PDF/圖片)
    ↓
模板比對：辨識文件類型 (enterprise_1 / contractor_1 / contractor_2)
    ↓
幾何校正：透視變換對齊至模板尺寸
    ↓
ROI 擷取：從對齊後影像裁切各欄位區域
    ↓
AIP 前處理：與空白模板做像素差異比對，判斷欄位是否有內容
    ↓
VLM 辨識：透過 Ollama glm-ocr 模型提取文字內容
    ↓
輸出：JSON 結果 + CSV 匯出 + 視覺化標註圖
```

### 模組功能一覽

| 模組 | 功能 |
|------|------|
| `main.py` | 程式入口，掃描巢狀目錄並執行完整流程 |
| `pipeline.py` | 文件處理管線：PDF 轉檔 → 特徵擷取 → 模板比對 → 對齊 → ROI 擷取 |
| `vlm_recognizer.py` | 對每個 ROI 執行 AIP + VLM 辨識，產生欄位結果 |
| `roi_preprocessor.py` | AIP 引擎：ECC 對齊 + BGR 差異計算，判斷有無內容 |
| `vlm_loader.py` | Ollama 連線管理，GPU 自動偵測 |
| `field_schema.py` | 欄位定義與提示詞模板（由 `update_configs.py` 自動產生） |
| `csv_exporter.py` | 將辨識結果匯出為 CSV |
| `output.py` | 儲存視覺化圖片、metadata JSON、ROI 圖片 |
| `update_configs.py` | 組態產生器：從 LabelMe 標註產生所有設定檔 |

---

## 環境需求

- **Python**: 3.9 以上
- **GPU**: NVIDIA GPU，建議 6GB 以上 VRAM
- **Ollama**: 需安裝並執行於 `localhost:11434`
- **模型**: `glm-ocr`（首次執行時會自動下載）

---

## 安裝步驟

### 1. 建立 Python 環境

```bash
conda create -n vlmcv python=3.9
conda activate vlmcv
```

### 2. 安裝套件

```bash
cd VLM-pdfRecognizer
pip install -r requirements.txt
```

### 3. 安裝 Ollama

前往 [https://ollama.com/download](https://ollama.com/download) 下載安裝。

### 4. 下載 VLM 模型

```bash
ollama pull glm-ocr
```

### 5. 產生設定檔（首次使用）

```bash
python update_configs.py
```

此步驟會產生：
- `data/{模板ID}/config.json` — ROI 座標設定
- `data/{模板ID}/blank_rois/*.png` — 空白 ROI 參考圖
- `vlm_pdf_recognizer/recognition/field_schema.py` — 欄位定義與提示詞

---

## 執行方式

### 啟動 Ollama 伺服器

在另一個終端機中執行：

```bash
ollama serve
```

### 放置待處理文件

將 PDF 文件依照以下結構放入 `input/` 目錄：

```
input/
└── 日期/
    └── 案件編號/
        ├── 文件1.pdf
        ├── 文件2.pdf
        └── ...
```

例如：
```
input/
└── 2026-02-24/
    └── case_a101/
        ├── doc1.pdf
        └── doc2.pdf
```

### 執行完整流程（AIP + VLM）

```bash
python main.py
```

### 僅執行 AIP（不啟用 VLM 辨識）

```bash
python main.py --disable-vlm
```

### 查看結果

結果輸出於 `output/` 目錄，結構對應 `input/`：

```
output/
└── 2026-02-24/
    ├── VLM_results.json                        # 該日期的完整辨識結果
    ├── vlm_recognition_results.csv             # 該日期的 CSV 格式匯出
    └── case_a101/
        ├── doc1_visualization.png              # 視覺化標註圖
        ├── metadata/doc1_metadata.json         # 處理元資料
        ├── rois/doc1_roi_person1.png           # 擷取的 ROI 圖片
        └── processed_rois/doc1_roi_person1_processed.png  # AIP 差異圖
```

#### 視覺化標註圖色碼說明

| 顏色 | 含義 |
|------|------|
| 綠色框 | 偵測到內容 |
| 紅色框 | 未偵測到內容 |
| 藍色框 | 無法判斷 |

---

## 新增模板操作

當需要支援新的文件類型時，依照以下步驟操作：

### 步驟 1：準備模板影像

- 取得一份**空白**（未填寫）的文件掃描檔
- 儲存為 JPG 格式至 `templates/images/{模板ID}.jpg`
- 模板ID 命名建議：使用英文底線格式，例如 `contractor_3`

### 步驟 2：使用 LabelMe 標註 ROI

安裝 LabelMe（如尚未安裝）：

```bash
pip install labelme
```

開啟標註工具：

```bash
labelme templates/images/{模板ID}.jpg --output templates/location/{模板ID}.json
```

標註規則：
- 使用 **矩形 (Rectangle)** 工具框選每個需要辨識的欄位
- **Label 命名規則**（會自動推斷欄位類型）：

| Label 包含的關鍵字 | 自動推斷類型 | 說明 |
|-------------------|-------------|------|
| `title` | title | 文件標題（預定義值，不做辨識） |
| `version` | version | 版本號碼 |
| `VX` 開頭 | checkbox | 勾選框 |
| `big1`, `small1` | stamp | 印章/公司章 |
| `person_number` | person_number | 身分證字號 |
| `number`, `year`, `month`, `date` | number | 數字欄位 |
| 其他 | text | 一般文字欄位 |

- 標註順序很重要：**LabelMe 中的標註順序必須與 field_schema.py 中的欄位順序一致**

### 步驟 3：註冊新模板

編輯 `vlm_pdf_recognizer/templates/template_loader.py`，在 `load_all_templates()` 函式中加入新模板 ID：

```python
def load_all_templates(data_dir="data", templates_dir="templates"):
    template_ids = ["enterprise_1", "contractor_1", "contractor_2", "contractor_3"]  # 加入新ID
    # ...
```

### 步驟 4：產生設定檔

```bash
python update_configs.py
```

此指令會自動產生：
- `data/{模板ID}/config.json`
- `data/{模板ID}/blank_rois/*.png`
- 更新 `field_schema.py`（包含新模板的欄位定義）

### 步驟 5：驗證

放入對應模板的測試文件，執行：

```bash
python main.py
```

檢查 `output/` 中的視覺化圖片，確認 ROI 框選位置正確。

---

## 修改現有 ROI 操作

當需要調整現有模板的 ROI 位置或新增/移除欄位時：

### 調整 ROI 位置

1. 開啟 LabelMe 編輯現有標註：

```bash
labelme templates/images/{模板ID}.jpg --output templates/location/{模板ID}.json
```

2. 拖曳或調整矩形框位置
3. 儲存後重新產生設定檔：

```bash
python update_configs.py
```

### 新增 ROI 欄位

1. 在 LabelMe 中新增矩形標註
2. 命名 Label（參照上方命名規則）
3. 儲存並執行 `python update_configs.py`

### 移除 ROI 欄位

1. 在 LabelMe 中刪除對應的標註
2. 儲存並執行 `python update_configs.py`

> **重要提醒**：每次修改 LabelMe 標註後，**務必執行 `python update_configs.py`** 重新產生設定檔。此指令會同步更新 `config.json`、`blank_rois/` 和 `field_schema.py`。

---

## 自訂 VLM 提示詞

VLM 對每種欄位類型使用不同的提示詞。如需修改提示詞內容：

1. 編輯 `update_configs.py` 中的 `PROMPT_TEMPLATES` 字典（約第 305-453 行）
2. 執行 `python update_configs.py` 重新產生 `field_schema.py`

目前支援的提示詞類型：

| 類型 | 用途 |
|------|------|
| `version` | 辨識 8 位版本號碼 |
| `checkbox` | 偵測勾選記號 |
| `stamp` | 辨識印章/公司章 |
| `text` | 擷取繁體中文文字 |
| `number` | 擷取純數字 |
| `person_number` | 辨識身分證字號（1 英文字 + 9 數字） |

> **注意**：請勿直接編輯 `field_schema.py`，此檔案會被 `update_configs.py` 覆蓋。

---

## 目錄結構說明

```
VLM-pdfRecognizer/
├── main.py                    # 程式入口
├── update_configs.py          # 設定檔產生器（必要時才執行）
├── requirements.txt           # Python 套件清單
│
├── templates/                 # 模板相關檔案
│   ├── images/                # 模板影像 (.jpg)
│   │   ├── enterprise_1.jpg
│   │   ├── contractor_1.jpg
│   │   └── contractor_2.jpg
│   └── location/              # LabelMe 標註檔 (.json)
│       ├── enterprise_1.json
│       ├── contractor_1.json
│       └── contractor_2.json
│
├── data/                      # 自動產生的設定（勿手動修改）
│   └── {模板ID}/
│       ├── config.json        # ROI 座標
│       ├── template_features.pkl  # SIFT 特徵快取
│       └── blank_rois/        # 空白 ROI 參考圖
│
├── input/                     # 放置待處理文件
│   └── {日期}/{案件編號}/*.pdf
│
├── output/                    # 處理結果輸出
│
└── vlm_pdf_recognizer/        # 主程式模組
    ├── pipeline.py            # 文件處理管線
    ├── output.py              # 結果輸出管理
    ├── preprocessing/         # PDF 轉檔
    ├── alignment/             # 文件對齊
    ├── extraction/            # ROI 擷取
    ├── recognition/           # AIP + VLM 辨識
    └── templates/             # 模板管理
```

---

## 常見問題

### Ollama 伺服器無法連線

```
RuntimeError: Ollama server unreachable
```

**解決方式**：確認 Ollama 正在執行：

```bash
ollama serve
```

### 模型未找到

首次使用時，需先下載模型：

```bash
ollama pull glm-ocr
```

### 文件無法比對到任何模板

```
UnknownDocumentError: inlier_count < 50
```

**可能原因**：
- 文件類型不在已註冊的模板中
- 掃描品質太差或角度偏差過大
- 文件不是此系統支援的文件類型

### 執行 update_configs.py 後欄位順序錯亂

**注意**：LabelMe 標註的順序即為 ROI 的處理順序。如果發現欄位對應錯誤，請在 LabelMe 中確認標註順序是否正確。

### GPU 記憶體不足

如 GPU VRAM 不足，可嘗試：
- 確認無其他程式佔用 GPU 記憶體
- 考慮使用較小的 VLM 模型

---

## 日常維護重點

1. **`data/` 目錄**內的檔案皆為自動產生，勿手動修改
2. **`field_schema.py`** 為自動產生，修改提示詞請編輯 `update_configs.py`
3. 每次修改模板標註後，**務必執行 `python update_configs.py`**
4. 確保 Ollama 伺服器在執行 `main.py` 前已啟動
5. `template_features.pkl` 為 SIFT 特徵快取，若模板影像更換需刪除此檔案讓系統重新計算
