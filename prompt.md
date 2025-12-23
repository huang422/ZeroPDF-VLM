# 角色定義 (Role Definition)
您是一位精通電腦視覺 (OpenCV) 與地端多模態模型 (Local VLM) 的 Python 開發專家。您的任務是實作一套地端、輕量化且無需訓練 (Zero-shot) 的文件處理系統，專門用於處理繁體中文的掃描 PDF 文件。

# 專案概述 (Project Overview)
目標是在資源受限的環境（純 CPU 或輕量級 GPU，如 4GB VRAM）下執行文件處理，優先使用GPU如果沒有GPU就用CPU。
工作流程包含三個主要階段：
1.  **分類與對齊 (Classification & Alignment)：** 自動辨識輸入的掃描檔屬於哪一種範本 (Template)，並透過幾何校正 (Warp) 將其與範本對齊。
2.  **提取 (Extraction)：** 根據範本預定義的座標，裁切出感興趣區域 (ROI)。

# 技術架構 (Technical Architecture)
專案資料夾結構清楚簡潔不要太複雜先依照此階段需求完成讓我驗證在進入下個推論階段。

## 1. 設定策略 (Configuration Strategy)
系統依賴「黃金範本 (Golden Templates)」。針對每一種文件類型data/ (enterprise_1, contractor_1, contractor_2)，我們儲存：
1.  一張乾淨的範本圖片（或預先計算好的 SIFT 特徵）。
2.  一個 JSON 設定檔，定義了該範本上的 ROI 固定座標，座標檔於/data中json檔，座標由左上到右下的bounding box。

使用python環境vlmcv
## 實作流程步驟 (Pipeline Steps)
1. 影像前處理 (去除浮水印與干擾)，移除
技術手段：OpenCV (HSV Thresholding / Adaptive Thresholding)
將圖片從 BGR 轉為 HSV 色彩空間。
提取黑色遮罩：設定閾值，只保留顏色深（Value 低）的像素。藍色、淺灰、淺紅浮水印的 Value 通常較高（較亮），會被濾除。
二值化 (Binarization)：將處理後的圖轉為純黑白（只有 0 和 255）。
預期結果：圖片中只剩下黑色的表格線、標題文字、以及使用者的黑色/深藍色原子筆跡/紅色印章。背景的藍色塊會變全白。

2. 文件對齊 (Alignment)
因為是掃描檔，會有歪斜或位移。技術手段：OpenCV (SIFT/ORB 特徵點匹配 + 透視變換)
準備 3 張「標準空白範本」圖片。
提取輸入圖片的特徵點，與範本進行匹配。
計算 Homography 矩陣，將輸入圖片「拉正」，使其座標與範本完全重疊。

步驟 1：範本匹配與對齊 (Router & Alignment)
演算法： SIFT (Scale-Invariant Feature Transform) + Homography (單應性矩陣)。
邏輯：
提取輸入圖片的 SIFT 特徵點與描述子。
使用 cv2.FlannBasedMatcher 或 BFMatcher 將其與所有預載的範本進行特徵匹配。
投票機制 (Vote)： 擁有最多「良好匹配點 (Good Matches / Inliers)」的範本即為目標文件類型。
幾何校正 (Warp)： 使用匹配點計算 Homography 矩陣 (
H
H
)。務必使用 RANSAC 演算法來排除雜訊（如手寫筆跡造成的干擾）。最後使用 cv2.warpPerspective 將輸入圖校正至範本座標系。
步驟 2：ROI 裁切與拼圖優化 (Stitching Optimization)
邏輯：
根據匹配到的設定檔座標，從「已校正」的輸入圖中裁切出各個 ROI，將輸入資料的圖片校正過後將校正和對應的bounding box呈現在輸出的以校正圖片上讓我做驗證，路徑存放在output。
儲存三個Golden Templates的結果和矩陣，這樣就不用每次都運算一次直接比對就好。
加入錯誤處理機制（例如：若匹配點數量小於 50，拋出 "Unknown Document" 例外）。
程式碼和註解都使用英文

/speckit.constitution	Create or update project governing principles and development guidelines
/speckit.specify	Define what you want to build (requirements and user stories)
/speckit.plan	Create technical implementation plans with your chosen tech stack
/speckit.tasks	Generate actionable task lists for implementation
/speckit.implement	Execute all tasks to build the feature according to the plan



3.  **推論 (Inference)：** 使用輕量級 VLM 一次性讀取ROI標題並偵測簽名或打勾的狀態。(稍後提供prompt需求再執行)

# 限制與需求 (Constraints & Requirements)
*   **硬體限制：** 必須能在 CPU 或消費級 GPU 上運行，需極大化執行效率。
*   **隱私要求：** 完全離線執行 (Offline)，不可呼叫外部 API。
*   **語言環境：** 繁體中文 (Traditional Chinese)。
*   **無訓練資料：** 必須採用 Zero-shot 方案（規則式視覺算法 + 預訓練 VLM）。
*   **強健性：** 必須能處理掃描雜訊（歪斜、旋轉、浮水印）。


拼圖 (Stitching)： 將所有裁切下來的小圖「垂直堆疊」成一張長條圖。
視覺分隔： 在小圖之間插入一條黑色水平線 (高度 5-10px)，幫助 VLM 區分不同區塊。
目的： 這讓我們能透過 單次 VLM 推論 完成所有任務，大幅減少 CPU 運算時間。
步驟 3：VLM 推論 (Zero-Shot Inference)
模型選擇： OpenGVLab/InternVL2_5-1B (首選，因其 OCR 能力強且極輕量) 或 Qwen/Qwen2-VL-2B-Instruct。
量化： 如果是 CPU 環境，請使用 INT4 或 INT8 量化載入模型。
提示詞工程 (Prompt Engineering)：
"這張圖片包含 {N} 個不同的區域，由黑線分隔。
從上到下依序為：
[標題區]：請輸出文字內容。
[簽名區]：請問是否有手寫簽名或蓋章？(請回答 Yes 或 No)
[打勾區]：請問是否有打勾？(請回答 Yes 或 No)
請以純 JSON 格式回傳結果。"
程式碼生成需求 (Code Generation Request)
請生成完整的 Python 專案結構程式碼，包含：
ImageAligner Class：
包含載入範本並快取 SIFT 特徵的方法。
方法 align_image(input_img)：需回傳匹配的文件類型名稱以及校正後的圖片。
VLMInference Class：
包含載入地端模型與 Tokenizer 的方法 (需考慮 CPU 優化)。
方法 predict(stitched_image, prompt)：處理多模態對話並回傳結果。
Pipeline Class：
串接流程：載入 -> 對齊 -> 裁切與拼圖 -> VLM -> JSON 輸出。
main.py：
範例執行腳本。
重要實作細節：
影像處理請使用 opencv-python。
模型載入請使用 transformers。
加入錯誤處理機制（例如：若匹配點數量小於 50，拋出 "Unknown Document" 例外）。
程式碼必須模組化且附帶詳細註解。

The primary workflow is as follows:
1.  **Batch Processing**: The `main.py` script serves as the entry point, reading all documents from the input directory.
2.  **Pipeline Execution**: For each document, the `DocumentProcessor` orchestrates a multi-step CV pipeline.
3.  **Feature Extraction**: Extracts SIFT features from the input document.
4.  **Template Matching**: Compares the document's features against a cache of pre-computed template features to find the best match.
5.  **Geometric Correction**: Warps the input document to align it perfectly with the matched template, correcting for perspective distortions.
6.  **ROI Extraction**: Crops specified regions from the aligned document based on coordinates defined in the template's configuration.

### Module Descriptions
- **`main.py`**: The main application entry point. Manages file I/O and drives the batch processing loop by invoking the document processor for each input file.

- **`vlm_pdf_recognizer/pipeline.py`**: The heart of the application. Contains the `DocumentProcessor` class which defines and executes the entire CV pipeline from start to finish.

- **`vlm_pdf_recognizer/alignment/`**: A module containing the core computer vision algorithms for the alignment stage.
 `feature_extractor.py`: Responsible for extracting SIFT keypoints and descriptors from images.
 `template_matcher.py`: Implements the logic to compare feature sets between the input and templates to identify the best match.
 `geometric_corrector.py`: Calculates the perspective transformation matrix and applies it (`cv2.warpPerspective`) to align the document.

- **`vlm_pdf_recognizer/templates/`**: Manages the loading and structure of template data.
 `template_loader.py`: Loads template images, their corresponding ROI configurations from JSON files, and the pre-computed SIFT feature caches (`.pkl`).

- **`vlm_pdf_recognizer/extraction/`**: Handles the final data extraction step.
 `roi_extractor.py`: Contains the logic to crop the ROIs from the now-aligned document image using the coordinates provided by the matched template.

- **`update_configs.py`**: A utility script used for project setup. Its likely purpose is to pre-compute and cache the SIFT features for all golden templates, optimizing the main pipeline's performance.