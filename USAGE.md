# 使用說明

## 快速開始

### 1. 安裝依賴（只需執行一次）

```bash
pip install -r requirements.txt
```

### 2. 生成配置檔案（只需執行一次）

```bash
python update_configs.py
```

### 3. 放入要處理的文件

將您的圖片或 PDF 檔案放到 `input/` 資料夾：

```bash
cp /path/to/your/documents/*.jpg input/
```

### 4. 執行處理

**預設啟用 VLM 辨識**：
```bash
python main.py
```

**停用 VLM（僅前處理）**：
```bash
python main.py --disable-vlm
```

## 輸出結果

處理完成後，在 `output/` 資料夾中會看到：

- `{檔名}_visualization.png` - 帶有 ROI 邊界框的視覺化圖片（VLM 模式下會顯示辨識結果）
- `{檔名}_metadata.json` - 處理資訊（匹配的模板、信心分數等）
- `VLM_results.json` - 批次處理總結（整合前處理和 VLM 辨識結果）
- `vlm_recognition_results.csv` - VLM 辨識結果 CSV 檔（僅 VLM 模式）

## 目錄結構

```
VLM-pdfRecognizer/
├── templates/              # 模板檔案
│   ├── images/            # 模板圖片（enterprise_1.jpg 等）
│   └── location/          # ROI 座標標註（*.json）
├── data/                  # 自動生成的配置
│   ├── enterprise_1/
│   │   ├── config.json
│   │   └── template_features.pkl
│   └── ...
├── input/                 # 放入要處理的文件
├── output/                # 處理結果輸出
├── main.py               # 主程式
└── update_configs.py     # 配置生成工具
```

## 支援的檔案格式

- 圖片：.jpg, .jpeg, .png
- PDF：.pdf（會自動轉換每一頁為圖片）

## 模板說明

系統預設有三個模板：

1. **enterprise_1** - 企業文件模板 1（17 個 ROI）
2. **contractor_1** - 承包商文件模板 1（15 個 ROI）
3. **contractor_2** - 承包商文件模板 2（2 個 ROI）

程式會自動識別輸入文件屬於哪個模板。

## 常見問題

### Q: 如何新增或修改 ROI？

1. 使用 LabelMe 工具標註 `templates/images/{template_id}.jpg`
2. 將標註結果儲存為 `templates/location/{template_id}.json`
3. 執行 `python update_configs.py` 重新生成配置

### Q: 處理速度如何？

- CPU 模式：每個文件約 2-3 秒
- 第一次執行會計算並快取 SIFT 特徵，之後會快 40%

### Q: 如何查看處理結果？

查看 `output/{檔名}_metadata.json` 來了解：
- 匹配到哪個模板
- 信心分數（inlier 數量）
- 處理時間
- 提取的 ROI 清單

### Q: 如果文件無法識別怎麼辦？

檢查 metadata.json 的 `error_message` 欄位，可能原因：
- 文件與任何模板都不匹配（confidence_score < 50）
- 圖片品質過低
- 浮水印太多影響特徵提取

## 進階設定

### 修改匹配閾值

編輯 `vlm_pdf_recognizer/alignment/template_matcher.py`，修改：

```python
MIN_INLIERS = 50  # 最小匹配特徵點數量
```

### 修改 SIFT 特徵數量

編輯 `vlm_pdf_recognizer/alignment/feature_extractor.py`，修改：

```python
nfeatures=5000  # 文件特徵點數量上限
```
