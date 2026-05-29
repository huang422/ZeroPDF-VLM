# ZeroPDF-VLM 開發流程與 PM 規格管理 (Development Workflow)

**Audience**: 想參與 / 維護本專案的工程師、PM、tech lead
**Last Updated**: 2026-05-26

這份文件回答兩個問題：

1. **業界標準的軟體 PM 開發流程是什麼？** — 對應到 PRD、User Story、Spec、Plan、Tasks 等通用概念。
2. **以 ZeroPDF-VLM 為例，怎麼套用這套流程？** — 包含目前專案採用的 GitHub Spec-Kit 工作流、四個 feature spec 的層級、以及實務上開新需求 / 改動既有功能的 SOP。

---

## Part 1 — 業界標準 PM 開發流程速覽

軟體開發 PM 在開發前期會建立的文件，由戰略層往執行層走：

| 層級 | 文件 | 回答的問題 | 主要受眾 |
|---|---|---|---|
| **戰略層** | Vision / Strategy | 為什麼要做？市場機會？ | C-level、投資人 |
| **產品層** | **PRD** (Product Requirements Doc) | 產品要解決什麼？有哪些功能？ | PM、設計、工程 leader |
| **使用者層** | **User Story** / Use Case / Persona | 誰會怎麼用？ | 設計、QA、工程師 |
| **設計層** | **Spec** / Functional Spec / Design Doc | 系統要做到什麼？模組怎麼切？ | 工程師 |
| **架構層** | **TDD** (Technical Design Doc) / RFC | API、Data Model、流程圖、技術選型 | 資深工程師、Tech Lead |
| **執行層** | **Tasks** / Backlog / Tickets | 誰做什麼？什麼時候做完？ | 整個團隊 |

### PRD 標準結構

```
1. Overview / Background          — 背景、為何而做
2. Goals & Non-Goals              — 明確列「不做什麼」
3. Success Metrics (KPI)          — 怎麼衡量成功
4. User Personas                  — 主要使用者輪廓
5. User Stories / Use Cases       — 使用者怎麼用
6. Functional Requirements (FR)   — 必須做的功能（FR-1, FR-2…）
7. Non-Functional Requirements    — 效能、安全、可用性
8. Out of Scope                   — 明確排除項
9. Open Questions / Assumptions   — 待釐清
10. Milestones / Timeline         — 時程
11. Risks                         — 風險與緩解
```

### User Story 寫法（INVEST 原則）

INVEST = **I**ndependent、**N**egotiable、**V**aluable、**E**stimable、**S**mall、**T**estable。

標準格式：
```
As a <角色>,
I want <功能/動作>,
so that <價值/目的>.

Acceptance Criteria:
  Given <前提條件>
  When <執行動作>
  Then <預期結果>
```

範例：
```
As a 文件處理人員,
I want 批次上傳掃描的 PDF 並自動識別欄位內容,
so that 我不需要逐筆人工輸入資料.

Acceptance Criteria:
  Given 一個含 3 種模板類型的案件資料夾,
  When 我執行 main.py,
  Then 每個 PDF 應產出 JSON 結果且案件層級驗證結果正確.
```

### 業界常見開發流程框架

| 框架 | 適合場景 | 文件節奏 |
|---|---|---|
| **Waterfall** | 規格穩定（金融、醫療系統） | 一次寫完整 PRD/Spec |
| **Agile / Scrum** | 需求變動快（多數 SaaS） | Epic → Story → Sprint Task |
| **Shape Up** (Basecamp) | 小團隊、避免無限重做 | 6 週 Cycle + Pitch |
| **Spec-Driven Development** | AI/工具導向、文件即程式起點 | Constitution → Spec → Plan → Tasks → Code |

**本專案採用最後一類** — GitHub Spec-Kit 風格的 Spec-Driven Development。

---

## Part 2 — ZeroPDF-VLM 的實際 PM 流程

### 2.1 文件層級對照

| 業界概念 | 本專案位置 | 角色 |
|---|---|---|
| Vision / 公司策略 | _(無 — 個人專案，戰略放在 README.md 的 "Overview" 段)_ | — |
| **Constitution** | [`.specify/memory/constitution.md`](.specify/memory/constitution.md) | 7 大核心原則：Local-First、Zero Training、Deterministic Validation、Graceful Degradation、Spec-Code Coherence、Simplicity、Traditional Chinese First |
| PRD (產品層) | [`README.md`](README.md) | 對外的 product overview + pipeline architecture + features + 系統需求 |
| **Feature Spec** (設計層) | `specs/<NNN-feature-name>/spec.md` | 每個 feature 的功能規格：User Stories、FR、SC、Assumptions、Out of Scope |
| **Plan / TDD** (架構層) | `specs/<NNN-feature-name>/plan.md` | 技術設計：選型、模組切分、複雜度說明 |
| **Research** | `specs/<NNN-feature-name>/research.md` | 技術選型的研究紀錄 + 對比方案 |
| **Data Model** | `specs/<NNN-feature-name>/data-model.md` | dataclass / 資料結構參考 |
| **Quickstart** | `specs/<NNN-feature-name>/quickstart.md` | 怎麼啟動 / 驗證這個 feature |
| **Tasks** (執行層) | `specs/<NNN-feature-name>/tasks.md` | 拆解後的可執行任務清單（歷史用） |
| Code | `vlm_pdf_recognizer/`、`main.py`、`update_configs.py` | 真正的實作 |

### 2.2 四個 Feature Spec 的分工

```
constitution.md
      │
      ▼
README.md (product overview)
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│ Feature 001 — Document Template Alignment & ROI Extraction   │
│   PDF → SIFT → FLANN+RANSAC → homography warp → ROI crop     │
│   產出: List[ExtractedROI], aligned image                     │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ Feature 004 — AIP (ROI Content Detection)                    │
│   ECC sub-pixel align → BGR mean diff → 2-tier threshold     │
│   產出: has_content per field (deterministic)                 │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ Feature 002 — VLM-Based ROI Content Recognition              │
│   Ollama glm-ocr → 每個非空白欄位 OCR → 文字後處理             │
│   驗證 (VX1 / date OR / others AND incl. VX2)                │
│   案件層級聚合 (3 模板都到齊)                                  │
│   產出: VLM_results.json + CSV + result_log.md + 視覺化       │
└──────────────────────────────────────────────────────────────┘

Feature 003 — VLM Auxiliary ROI Comparison
   ⚠️ 已被 Feature 004 取代 (SIFT-on-ROI 方案在小欄位不可行)
   保留作為歷史紀錄
```

### 2.3 Spec-Kit 工作流（這個專案目前用的）

```
/speckit.constitution  → 寫專案的核心原則 (一次性，鮮少修改)
       ↓
/speckit.specify       → spec.md (what & why, 不寫 how)
       ↓
/speckit.clarify       → 找出規格的曖昧處並補上
       ↓
/speckit.plan          → plan.md (技術設計、架構、模組切分)
       ↓
/speckit.tasks         → tasks.md (拆解為可執行小任務)
       ↓
/speckit.implement     → 按 tasks 寫程式碼
       ↓
/speckit.analyze       → 一致性檢查 (spec/plan/tasks 是否對齊)
```

每個 feature 一個資料夾，產出至少 6 個檔案：`spec.md`、`plan.md`、`research.md`、`data-model.md`、`tasks.md`、`quickstart.md`，外加 `contracts/`、`checklists/`。

---

## Part 3 — 實務 SOP

### 3.1 開新 feature 的標準流程

例如要新增 "PDF 浮水印自動辨識" 這個 feature：

1. **建立新 feature 資料夾**：`specs/005-watermark-detection/`，編號遞增。
2. **寫 `spec.md`**（功能規格）：
   - 用 User Story 描述使用者場景與優先級 (P1/P2/P3)
   - 列出 FR-001, FR-002... 的具體需求
   - 列出 Success Criteria (可量化)
   - 列出 Assumptions（依賴的前置條件）
   - 列出 Out of Scope（**這個很重要，明確排除不做的事**）
3. **寫 `plan.md`**（技術設計）：
   - Technical Context（語言、套件、硬體）
   - Constitution Check（檢查是否違反 7 大原則）
   - Project Structure（檔案位置）
   - Phase 0 (Research) / Phase 1 (Design) / Phase 2 (Tasks) 規劃
4. **寫 `data-model.md`**：列出 dataclass、entity 關係。
5. **寫 `quickstart.md`**：怎麼測試這個 feature。
6. **寫 `tasks.md`**：拆解為 ≤ 1 天可完成的任務（如 T001、T002...）。
7. **實作**：按 `tasks.md` 順序開始寫程式碼。
8. **每完成一個 FR 或 task**：
   - 標記 task 完成
   - 確認 spec / data-model 仍然描述正確的行為
   - 若行為與 spec 偏差，**先改 spec 再改 code**（憲法 V 條款）

### 3.2 改動既有 feature 的標準流程

例如要把 VX2 變成 "可選" 而不是必選：

1. **判斷影響的 feature**：VX2 邏輯在 Feature 002 (`calculate_results_status`)。
2. **先改 spec**：`specs/002-vlm-roi-recognition/spec.md` 的 FR-012。
3. **更新 `Last Aligned With Code` 日期**。
4. **新增 "Drift from previous behaviour" 註記**，說明為什麼改、何時改。
5. **改 code**：`vlm_pdf_recognizer/recognition/vlm_recognizer.py:calculate_results_status`。
6. **改 data-model.md** 中對應演算法描述。
7. **改 README.md** 中 "Document Validation Logic" 段落。
8. **PR 描述**：列出影響範圍（spec、code、README）。
9. **跑一次代表性的 batch** 確認 `result_log.md` 行為符合預期。

### 3.3 改動參數 / 閾值的標準流程

例如要把 AIP `MIN_ABSOLUTE_DENSITY_THRESHOLD` 從 0.01 降為 0.005：

1. **判斷影響的 feature**：Feature 004 (`roi_preprocessor.py`)。
2. **改 spec.md FR-003 / FR-010**：把新值寫進去。
3. **改 data-model.md** 的常數表。
4. **改 code**：`roi_preprocessor.py` 的 `MIN_ABSOLUTE_DENSITY_THRESHOLD`。
5. **跑兩組測試 batch**：分別用舊值跟新值。
6. **PR 描述附上**：兩組批次的 `result_log.md` 對比，說明改善/退化在哪。
7. **若退化嚴重 → 回滾**；若改善 → merge 並更新 README.md 的 Configuration & Tuning 段。

### 3.4 新增 template 類型的標準流程

例如要新增 `partner_1` template：

1. **準備素材**：
   - `templates/images/partner_1.jpg`（乾淨的金樣）
   - `templates/location/partner_1.json`（LabelMe 標註）
2. **更新 template loader**：
   - `vlm_pdf_recognizer/templates/template_loader.py:load_all_templates()` 加上 `partner_1`。
3. **跑配置產生器**：`python update_configs.py`
   - 會自動產生 `data/partner_1/config.json`、`data/partner_1/blank_rois/*.png`、更新 `field_schema.py`。
4. **判斷是否進入 `REQUIRED_TEMPLATE_TYPES`**：
   - 如果這個新模板要納入「案件必須三個模板齊備」的判斷 → 改 `vlm_pdf_recognizer/output.py:202` 的 `REQUIRED_TEMPLATE_TYPES`。
   - 同時更新 `specs/002-vlm-roi-recognition/spec.md` FR-013。
5. **準備測試 PDF** 放到 `input/<date>/<case_id>/` 跑一次。
6. **檢查所有輸出**：`VLM_results.json`、CSV、視覺化、`result_log.md`。
7. **更新 README.md 的 "Adding New Templates" 段**，記錄這個 template 的特殊欄位 / 行為。

### 3.5 Spec 與 Code 出現漂移時的修復 SOP

當你發現 spec 描述跟 code 實際行為不一致：

1. **判斷哪個是正確的**：
   - **如果是 code bug** → 改 code，spec 不動。
   - **如果是 spec 過時** → 改 spec，加上 "Drift from previous draft" 段落紀錄歷史。
2. **如果 spec 提到的 entity / function 在 code 中不存在**：
   - 在 spec 加上 `Status: SUPERSEDED` header，連到取代它的 spec / file。
3. **如果 code 有 spec 沒提的功能**：
   - 補上對應的 FR + Acceptance Scenarios。
4. **更新所有受影響檔案的 `Last Aligned With Code` 日期**。

> 這個專案在 2026-05-26 做過一次大規模的 spec-code 對齊（包含 InternVL→Ollama 換軌、6 步 AIP→2 步 AIP 簡化、case 層級驗證新增、VX2 必選規則）。詳見各 spec 的 "Drift from the original draft" 段落。

---

## Part 4 — 各角色職責對照

| 角色 | 主要文件 | 主要工具 |
|---|---|---|
| **Product Owner / PM** | `README.md`、`specs/*/spec.md` | Spec-Kit slash commands |
| **Tech Lead / Architect** | `specs/*/plan.md`、`specs/*/research.md` | Plan reviews |
| **工程師** | `specs/*/tasks.md`、`specs/*/data-model.md`、code | IDE、`update_configs.py`、`main.py` |
| **QA** | `specs/*/quickstart.md`、`result_log.md` | 跑代表性 batch、檢查 CSV |
| **Reviewer (code review)** | 全部 — 確認 PR 同時更新 spec 與 code | GitHub PR review |

---

## Part 5 — 常見錯誤與避免方式

| 錯誤 | 後果 | 避免方式 |
|---|---|---|
| 直接改 code，忘記改 spec | 下一個人讀 spec 跟 code 看到的行為不一樣，浪費 debug 時間 | PR 必須同時包含 spec + code 改動 |
| Spec 寫得過於詳細（提到具體 class 名稱、行號） | code refactor 時 spec 全面失效 | spec 描述 "做什麼" 而非 "在哪一行怎麼寫"；具體 class 放 `data-model.md` |
| Spec 寫得過於模糊（沒有 FR 編號、沒有 Acceptance Criteria） | 工程師無法驗證 / 寫測試 | 強制要求 FR-001…FR-NNN + Given/When/Then |
| 在 spec / plan / data-model 中複製貼上相同資訊 | 三邊互相不同步、無法判斷哪個是真相 | spec 描述 "什麼"、plan 描述 "怎麼"、data-model 列 schema；互相連結而非複製 |
| 為了「未來可能需要」加入大量配置選項 | 維護負擔、bug surface 增加，但實際沒人調 | 違反憲法 VI（Simplicity）；先用 module-level 常數，需要時再升級 |
| Feature 之間互相依賴但 spec 沒寫清楚 | 改動 Feature A 意外破壞 Feature B | 每個 spec 末尾的 "Notes for Spec-vs-Code Reviewers" 列出對外承諾 |

---

## Part 6 — 快速參考

### 跑完整 pipeline
```bash
ollama serve &
python main.py
```

### 跑單純 alignment 階段
```bash
python main.py --disable-vlm
```

### 看 AIP 的中間結果
```bash
DEBUG_ROI_PREPROCESSING=true python main.py
# 結果在 output/processed_rois/<document>/<field>/
```

### 加新欄位 / 改模板
```bash
# 1. 改 templates/location/<id>.json
# 2. 重新產生 config
python update_configs.py
# 3. 跑一次
python main.py
```

### 查特定案件失敗原因
```bash
cat output/<date>/result_log.md
```

### 看 batch summary
```bash
cat output/<date>/VLM_results.json | jq '.case_results'
```

---

## Part 7 — 延伸閱讀

- [GitHub Spec-Kit](https://github.com/github/spec-kit) — 本專案採用的規格驅動開發框架
- [Shape Up by Basecamp](https://basecamp.com/shapeup) — 6-week cycle / Pitch 方法
- [Atlassian PRD Template](https://www.atlassian.com/agile/product-management/requirements) — 業界 PRD 範本參考
- [Writing User Stories with INVEST](https://www.agilealliance.org/glossary/invest/) — INVEST 原則

---

**Maintainer**: Tom Huang
**Contact**: huang1473690@gmail.com
**Last Updated**: 2026-05-26
