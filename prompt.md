/speckit.constitution	Create or update project governing principles and development guidelines
/speckit.specify	Define what you want to build (requirements and user stories)
/speckit.plan	Create technical implementation plans with your chosen tech stack
/speckit.tasks	Generate actionable task lists for implementation
/speckit.implement	Execute all tasks to build the feature according to the plan

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


   1. 特徵提取 (Feature Extraction)
   2. 特徵匹配與模型估算 (Feature Matching & Model Estimation)
   3. 幾何校準 (Geometric Correction)

  整個流程遵循一個經典的計算機視覺模式：特徵提取 -> 匹配 -> 幾何驗證與校正。

  ---

  第一步：特徵提取 (Feature Extraction)

  檔案: vlm_pdf_recognizer/alignment/feature_extractor.py
  演算法: SIFT (Scale-Invariant Feature Transform)

  目標
  為一張圖片創建一個穩定的「數位指紋」。這個指紋不應該因為圖片被放大縮小、旋轉、或光線改變而產生巨大變化。這樣才能確保無論文件掃描得如何，我們都能認出它。

  演算法詳解 (`extract_sift_features` 函式)

   1. 灰階轉換: SIFT 演算法在單一通道的灰階影像上運作。如果輸入的圖片是彩色的 (3個通道: BGR)，程式會先將其轉換為灰階。

   2. SIFT 物件創建 (`cv2.SIFT_create`):
       * 程式會初始化一個 SIFT 偵測器。這裡有一個重要的細節：它對「範本圖片」和「輸入文件」使用了不同的參數。
           * 範本 (Template): nfeatures=0 (不限制特徵數量)，contrastThreshold=0.04。目標是從乾淨的標準範本中提取所有可能穩定、高品質的特徵點。
           * 文件 (Document): nfeatures=5000 (最多提取5000個最強的特徵點以提升效能)，contrastThreshold=0.03 (稍微降低對比度門檻，以應對掃描品質不佳、對比度較低的文件，從而能捕捉到更多特徵)。

   3. 特徵點偵測與描述子計算 (`sift.detectAndCompute`):
       * 偵測 (Detect): SIFT 會在不同尺度（影像金字塔）上掃描圖片，尋找在所有尺度和鄰近位置上都是最突出（最亮或最暗）的點。這些點就是「特徵點 (Keypoints)」。它們通常是物件的角點或獨特的紋理點。
       * 計算 (Compute): 對於每一個被偵測到的特徵點，SIFT 會計算一個「描述子 (Descriptor)」。
           * 描述子是一個包含 128 個數字的向量（float32 陣列）。
           * 它描述了特徵點周圍區域的梯度（光線變化方向和強度）分佈。
           * 這個 128 維的向量就是這個特徵點的「指紋」。因為它的計算方式是標準化的，所以對旋轉和光線變化不敏感。

  輸出: 最終，這個步驟會產出兩樣東西：
   * 一個 KeyPoint 物件列表，記錄了每個特徵點的位置、大小和方向。
   * 一個 Nx128 的 numpy 陣列，其中 N 是特徵點的數量，每一行就是一個特徵點的 128 維描述子。

  ---

  第二步：特徵匹配與模型估算 (Feature Matching & Model Estimation)

  檔案: vlm_pdf_recognizer/alignment/template_matcher.py
  演算法: FLANN Matcher + Lowe's Ratio Test + RANSAC

  目標
  找出輸入文件和某個範本之間所有對應的特徵點，並從這些對應關係中計算出一個能描述「如何將輸入文件變換成範本樣式」的數學模型（單應性矩陣）。

  演算法詳解

   1. 特徵匹配 (`match_features` 函式):
       * FLANN Matcher: 程式使用 cv2.FlannBasedMatcher，這是一種為了在高維度空間中（例如 SIFT 的 128 維）快速尋找最近鄰而優化的演算法。
       * k-NN 匹配: 對於輸入文件中的每一個描述子，FLANN 會在範本的所有描述子中尋找與它最相似的 2 個 (k=2)。
       * Lowe's Ratio Test (洛氏比率檢定): 這是過濾掉壞匹配的關鍵步驟。
           * 假設對於文件中的某個特徵點 A，在範本中找到了兩個最相似的點 B1 和 B2，它們的相似度距離分別是 d1 和 d2 (d1 < d2)。
           * 如果 d1 / d2 < 0.7（程式中的 ratio_threshold），那麼我們就認為 A 和 B1 是一個「好的匹配」。
           * 為什麼這麼做？ 如果一個匹配是明確的，那麼 B1 應該要比任何其他點 B2 都"近得多"。如果 B1 和 B2 距離差不多（比率接近 1.0），代表這個匹配很模糊，可能是錯誤的，應該被丟棄。

   2. 單應性矩陣估算 (`compute_homography_and_inliers` 函式):
       * 單應性矩陣 (Homography Matrix): 這是一個 3x3 的矩陣，它可以描述一個平面上的點如何被投影到另一個平面上。在這裡，它描述了「輸入文件平面」到「範本平面」的透視變換關係。
       * RANSAC (Random Sample Consensus, 隨機抽樣一致性): 即使經過 Lowe's Ratio Test，"好的匹配"中仍然可能存在錯誤的匹配（稱為離群點, Outliers）。RANSAC 的目標是從充滿雜訊的數據中找出一個可靠的模型。
           * a. 隨機抽樣: 隨機從「好的匹配」中選出 4 對匹配點（計算一個單應性矩陣最少需要 4 對點）。
           * b. 計算模型: 根據這 4 對點，計算出一個候選的單應性矩陣 H。
           * c. 內群點計數: 使用這個 H 矩陣，將輸入文件中的所有特徵點進行變換，看看它們變換後的位置與範本中對應的特徵點位置有多接近。如果距離在一個門檻內（程式中的 ransac_threshold=5.0
             像素），就稱之為「內群點 (Inlier)」。
           * d. 迭代與投票: 重複 a-c 步驟數百次，每一次都會得到一個候選的 H 和對應的內群點數量。
           * e. 最佳模型: 最終，RANSAC 會選擇那個擁有最多內群點的 H 作為最可靠的單應性矩陣。這個 H 和它的內群點就是這個步驟的輸出。

   3. 投票選出最佳範本 (`match_templates` 函式):
       * 程式會對所有範本重複上述 1 和 2 步驟。
       * 最後，擁有最多「內群點 (Inliers)」的那個範本被視為最佳匹配。如果最佳匹配的內群點數量少於 50，則認為無法識別文件，並拋出錯誤。

  ---

  第三步：幾何校準 (Geometric Correction)

  檔案: vlm_pdf_recognizer/alignment/geometric_corrector.py
  演算法: Perspective Warp

  目標
  使用前一步計算出的最佳單應性矩陣 H，將原始的輸入文件進行變換，使其視角與範本完全對齊。

  演算法詳解 (`warp_perspective` 函式)

   1. 透視變換 (`cv2.warpPerspective`):
       * 這是一個 OpenCV 函式，它接收三個主要參數：
           1. 原始輸入圖片 (image)。
           2. 3x3 的單應性矩陣 (homography)。
           3. 輸出的目標圖片尺寸（即範本圖片的寬和高）。
       * 函式會根據 H 矩陣，將原始圖片中的每一個像素重新對應到輸出圖片的新位置上。這實質上是在進行拉伸、旋轉和傾斜校正，以消除掃描時的透視變形。
       * interpolation=cv2.INTER_LINEAR 參數表示在像素重新定位時，如果新位置不是整數，則使用雙線性插值法來計算其顏色，使結果影像更平滑。

  輸出: 一張與範本尺寸完全相同、視角完全對齊的圖片。在這張校準後的圖片上，所有 ROI 的位置都是固定且已知的，可以直接進行切割和後續處理。

  3.  **推論 (Inference)：** 使用輕量級 VLM 一次性讀取ROI標題並偵測簽名或打勾的狀態。(稍後提供prompt需求再執行)

# 限制與需求 (Constraints & Requirements)
*   **硬體限制：** 必須能在 CPU 或消費級 GPU 上運行，需極大化執行效率。
*   **隱私要求：** 完全離線執行 (Offline)，不可呼叫外部 API。
*   **語言環境：** 繁體中文 (Traditional Chinese)。
*   **無訓練資料：** 必須採用 Zero-shot 方案（規則式視覺算法 + 預訓練 VLM）。
*   **強健性：** 必須能處理掃描雜訊（歪斜、旋轉、浮水印）。



**VLM 推論 (Zero-Shot Inference)**
1. 根據現有所有專案程式碼新增功能，加入vlm模型辨識，依照目前的架構新增修改，目前程式所有運作功能都要保持正常。
2. 實作VLM模型： OpenGVLab/InternVL 3.5-1B。量化： 如果是 CPU 環境，請使用 INT4 或 INT8 量化載入模型。優先使用gpu如果沒gpu就用cpu。
3. 將目前程式校準完成後裁切好的ROI圖像進入VLM辨識，目前程式已經可以輸出校準好圖像的所有ROI標籤與位置。
4. 讓vlm(InternVL 3.5-1B)辨識這張片綠匡中裁切下來的區域，辨識內容有標籤（直接匯入對應）、主標題內容（可直接對應輸出）{標籤：主標題內容, contractor_1_title : '企業負責人電信信評報告之使用授權書', contractor_2_title, '個人資料特定目的外用告知事項暨同意書', enterprise_1_title : '企業電信信評報告之使用授權書'}。

5. 其餘的roi讓VLM辨識區域內有無簽名打勾或蓋章，請辨識不同的位置要辨識的東西幫我產生給vlm的prompting詳細且高要求讓模型正確辨識且不要輸出不必要的內容，先辨識有無蓋章或簽名再辨識內容，如果內容辨識錯誤沒關係或無法辨識就顯示null，但有無定一定要精準的輸出true or false。
辨識VLM各種類說明，除標題之外其餘內容輸出分為：
-  有無辨識到內容True False （二元選擇）。
-  如果True有辨識到就輸出辨識的內容文字，如果沒辨識到False或有辨識到但無法辨識內容的文字就輸出null。

如果匹配類別是[contractor_1]
{contractor_1_title:不用辨識直接輸出對應的主標題內容, VX1: 辨識圓圈中是否有打勾Ｖ的筆畫或原子筆的筆畫，如果只有圈圈外匡沒有打勾筆畫代表沒有,person1: 辨識有無填寫文字筆跡可能為藍色或黑色筆畫，可能為印刷字或手寫字，如果是空白的代表沒有, company1: 辨識有無填寫文字筆跡可能為藍色或黑色筆畫，可能為印刷字或手寫字，如果是空白的代表沒有, VX2: 辨識圓圈中是否有打勾的筆畫或原子筆的筆畫，如果只有圈圈外匡沒有打勾筆畫代表沒有, company2: 辨識有無填寫文字筆跡可能為藍色或黑色筆畫，可能為印刷字或手寫字，如果是空白的代表沒有, company_number1:辨識是否有寫編號（數字）存在，可能為手寫或印刷數字，如果只有預設的黑色匡線則代表沒有, person2: 辨識有無填寫文字筆跡可能為藍色或黑色筆畫，可能為印刷字或手寫字，如果是空白的代表沒有, person_number1:辨識使否有寫編號（數字）存在，可能為手寫或印刷數字，如果只有預設的黑色匡線則代表沒有, year: 辨識是否有寫年份（數字）存在，可能為手寫或印刷數字, month: : 辨識是否有寫月份（數字）存在，可能為手寫或印刷數字, date: : 辨識是否有寫日期（數字）存在，可能為手寫或印刷數字: 辨識使否有寫年份（數字）存在，可能為手寫或印刷數字, big:辨識是否有正方形或圓形的蓋章可能為紅色或黑色，如果只有黑色外匡或中央的預設文字（負責人蓋章處）或浮水印的淺黑色文字筆跡，代表沒有}


如果匹配類別是[contractor_2]
{ontractor_1_title:不用辨識直接輸出對應的主標題內容, small:辨識是否有正方形或圓形的蓋章可能為紅色或黑色，如果只有黑色外匡或中央的預設文字（負責人蓋章處）或浮水印的淺黑色文字筆跡，代表沒有}

如果匹配類別是[enterprise_1]
{enterprise_1_title:不用辨識直接輸出對應的主標題內容, VX1: 辨識圓圈中是否有打勾的筆畫或原子筆的筆畫，如果只有圈圈外匡沒有打勾筆畫代表沒有, person1: 辨識有無填寫文字筆跡可能為藍色或黑色筆畫，可能為印刷字或手寫字，如果是空白的代表沒有, company1: 辨識有無填寫文字筆跡可能為藍色或黑色筆畫，可能為印刷字或手寫字，如果是空白的代表沒有, VX2: 辨識圓圈中是否有打勾的筆畫或原子筆的筆畫，如果只有圈圈外匡沒有打勾筆畫代表沒有,company2: 辨識有無填寫文字筆跡可能為藍色或黑色筆畫，可能為印刷字或手寫字，如果是空白的代表沒有, person2: 辨識有無填寫文字筆跡可能為藍色或黑色筆畫，可能為印刷字或手寫字，如果是空白的代表沒有, company_number:辨識是否有寫編號（數字）存在，可能為手寫或印刷數字，如果只有預設的黑色匡線則代表沒有, address: 辨識有無填寫文字地址可能為印刷字或手寫字，如果是空白的代表沒有, year: 辨識是否有寫年份（數字）存在，可能為手寫或印刷數字, month: : 辨識是否有寫月份（數字）存在，可能為手寫或印刷數字, date: : 辨識是否有寫日期（數字）存在，可能為手寫或印刷數字: 辨識使否有寫年份（數字）存在，可能為手寫或印刷數字, big:辨識是否有正方形或圓形的蓋章可能為紅色或黑色，如果只有黑色外匡或中央的預設文字（負責人蓋章處）或浮水印的淺黑色文字筆跡，代表沒有, small:辨識是否有正方形或圓形的蓋章可能為紅色或黑色，如果只有黑色外匡或中央的預設文字（負責人蓋章處）或浮水印的淺黑色文字筆跡，代表沒有}

6. output 輸出邏輯更新：輸出資料是更新目前輸出的視覺化結果和一份csv檔描述所有辨識結果，辮識有無回傳顯示True or False欄位名稱用英文, 回傳csv欄位範例{image_ID, results, type, title, 其他辮識結果} 
- 其中results是如果此影像樣本有出現False沒辮識到的內容，status就回傳False，只要判斷有無的True or Falsw，內容是什麼沒關係不用納入條件，全部都辨識到（有無都是True都有辮識到就顯示，result就顯示True）。
- 其中year, month, date有其中一個有辨識到， result就可以顯示True
- 如果VX1有辨識到打勾，代表直接不同意，那後面不管辨識內容如何就顯示False，因此VX1要謹慎辨識，通常沒打勾的機率會比較高一點
- output 的視覺化輸出圖片也在標籤旁邊顯示有無的True or False

7. 如果VLM在辮識時異常或記憶體不足則自動重啟後進行辮識，不能讓系統停住。


辨識需求和csv輸出內容有錯誤
1. 能辨識templateid就一定能mapping title，title :contractor_1_title : '企業負責人電信信評報告之使用授權書', contractor_2_title, '個人資料特定目的外用告知事項暨同意書', enterprise_1_title : '企業電信信評報告之使用授權書'
2. 只有文字的底方需要辨識內容，其他打勾的地方辨識有無就好
3. 邏輯是VX1有打勾result(True)就會是False
4. 檢查我的所有邏輯需求和prompt改得清詳細一點，整理以下內容精簡詳細的寫入prompt：

VLM輔助辨識功能
1. 詳細閱讀現在所有的程式碼和文件，不能遺失或修改錯誤現有的運作功能和邏輯。
2. 新功能要完美整合銜接到現有的程式碼中，不能有錯誤或冗餘。
3. update_configs.py除了現有功能是產生template的向量提供對齊之外，還要根據template/提供的座標檔進行才切出ROI後產個中template的ROI相量儲存於data（空白樣本roi向量）。
4. 在辨識新樣本要將ROI放入VLM辨識時，先把新樣本的ROi跟空白樣本的ROi做比對（類似第一階段樣本對齊辨識template方法），諾新樣本ROI跟空白樣本ROI有很高的相似度，則代表該樣本沒有填寫蓋章或簽名。如果有差異代表可能有填寫蓋章或簽名，再送入VLM辨識。
5. 輸出維持現有的VLM辨識所有ROi的輸出，新增一欄輔助辨識的比對結果。
6. 輸出邏輯維持現有一樣，只是將result的判斷改為輔助辨識的比對結果優先，因為result目前只看有無不看內容，邏輯要跟現在一樣(現有邏輯是：VX1 True則result直接True，其餘所有內容應該都要是True，如果有False則result是False(除了year, month, date三個用or判斷如目前的邏輯，請先確認目前邏輯))
7. 視覺化的輸出結果也是改成用輔助辨識的結果True or False呈現紅色和綠色
8. VX1, VX2辨識checkbox維持現有方式

VX1/VX2（勾選框）：
aux=None（跳過輔助）
has_content=啟發式結果
需要用has_content（因為沒有aux）

一般欄位（有輔助比對）：
aux=True(分數低因為有填寫內容預控白樣本特徵不像)
has_content=aux
content_text=vlm讀取的內容
只用aux（忽略VLM）

空白欄位無特徵：
aux=分數很高因為跟空白很接近所以回傳False
has_content=False（因為空白沒填寫，不需要vlm確認）
需要用has_content

所有roi都會給vlm讀但視覺化呈（判斷綠紅）和判斷json result只用aux結果（因為只判斷有無）
json要有先前完整的資訊被你刪掉了，必須包含aux有無、vlm有無、vlm判斷的內容文字