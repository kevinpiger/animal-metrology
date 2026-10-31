# 現有 API 清單

更新：2026-11-01。依 `metrology/web.py` 與模型服務核對。

Docker 預設入口：`http://127.0.0.1:8080`。目前沒有登入驗證。

| 方法 | 路徑 | 成功回應 | 用途 |
|---|---|---|---|
| GET | `/api/health` | 200 JSON | 查詢 Seg／Pose 服務健康與裝置；查看 ready、models、runtime、scope、max_upload_mb |
| GET | `/api/samples` | 200 JSON 陣列 | 範例 id、name、file、animal_count、url、source_url |
| POST | `/api/jobs` | 202 JSON | 建立當次分析，回傳 id 與 status=queued |
| GET | `/api/jobs/{job_id}` | 200 JSON | 查詢當次工作進度；完成時包含 result，失敗時包含 error |
| GET | `/api/jobs/{job_id}/artifacts/{name}` | 200 檔案 | 下載當次分析產物 |

## 建立分析的參數

Content-Type：`multipart/form-data`。

| 欄位 | 類型 | 預設／限制 |
|---|---|---|
| sample_id | 字串 | 1818（斑馬）、5992（羊）、7281（馬）；與 image 二擇一 |
| image | 檔案 | JPEG／PNG／WebP；10 MiB、2,000 萬像素以內；與 sample_id 二擇一 |
| conf | 數值 | 0.25；0 ≤ 值 ≤ 1；兩模型偵測門檻 |
| eye_conf | 數值 | 0.5；0 ≤ 值 ≤ 1；眼點信心門檻 |
| match_iou | 數值 | 0.3；0 < 值 ≤ 1；跨模型配對門檻 |
| device | 字串 | 預設 cpu（依 MODEL_DEVICE）；介面仍接受 auto／cuda／mps／cpu，本次部署強制 CPU |

目前同時僅接受一筆未完成工作，非持久化工作佇列。狀態依序 queued → running → complete 或 failed。202 表示工作已接受，並非推論成功，仍須查詢狀態。

## 下載產物

允許名稱：`source.jpg`、`overlay.jpg`、`result.json`、`measurements.csv`、`animal_N_mask.png`（N 為動物 ID）。source.jpg 在工作建立後即可取用，其他產物須等工作完成。動物 Mask 不等於眼睛 Mask，現階段沒有眼睛遮罩。

JSON 結果包含原圖尺寸、裝置、模型、門檻、動物分割與配對、眼點合法性及原因、雙眼距離與跨動物右眼距離。距離單位為 px；無效距離為 null，CSV 對應空值。

## 錯誤與限制

| 情況 | 行為 |
|---|---|
| 同時有 queued／running 工作 | 409 |
| 檔案或像素數超標 | 413 |
| 未選輸入、兩種輸入同時提供、範例不存在、影像解碼失敗、參數不合法 | 422 |
| 工作不存在、產物未就緒、名稱不允許或檔案不存在 | 404 |
| 背景推論失敗 | 工作 status=failed，error 說明錯誤 |

Docker health 會查詢兩個模型服務。模型啟動時載入權重並暖機後才回報 ready；此健康狀態不等同正式準確率評估。ready=false 仍可能是 HTTP 200，部署健康檢查會解析回應內容。原生入口只檢查權重存在與張量運算。

## 呼叫流程

```bash
# 1. 查範例
curl http://127.0.0.1:8080/api/samples
# 2. 選範例分析；上傳時改用 -F image=@/path/to/image.jpg
curl -X POST http://127.0.0.1:8080/api/jobs -F sample_id=5992 -F device=auto
# 3. 將回傳的 id 代入並查詢，直到 complete 或 failed
curl http://127.0.0.1:8080/api/jobs/WORK_ID
# 4. complete 後下載
curl -o measurements.csv http://127.0.0.1:8080/api/jobs/WORK_ID/artifacts/measurements.csv
```

## 頁面及文件路由

| 路徑 | 用途 |
|---|---|
| `/` | 操作首頁 |
| `/static/{path}` | 網頁資源 |
| `/samples/{path}` | 範例圖片 |
| `/docs` | Swagger UI |
| `/redoc` | ReDoc |
| `/openapi.json` | OpenAPI 規格 |

[Swagger](http://127.0.0.1:8080/docs) · [ReDoc](http://127.0.0.1:8080/redoc) · [OpenAPI](http://127.0.0.1:8080/openapi.json)

## 不提供歷史紀錄

沒有工作列表、歷史搜尋、歷史報表或登入 API。job_id 用於當次分析及下載。目標為結果暫存 30 分鐘後清除；已實作 TTL 與背景清理；完成／失敗後開始計時，到期移除狀態及輸出。

Docker 的 Seg／Pose 分別提供內部 GET /health 與 POST /predict；POST 接收 multipart image、device、conf，回傳 names、boxes、keypoints、masks（Base64 PNG）、runtime。僅 Docker 內部使用，Web 不代理這兩個介面。原生入口保留本地函式推論。
