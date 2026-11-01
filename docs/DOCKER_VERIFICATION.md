# Docker 與服務驗證紀錄

更新：2026-11-01。

## 執行環境

- Docker／Colima：Linux ARM64，4 CPU、約 4 GiB 記憶體。
- Web、API、Seg、Pose 四容器已實際啟動，API 與兩個模型服務健康檢查通過。
- PyTorch 2.15.0+cpu，OpenCV 4.11.0；CUDA 不可用，兩模型明確使用 CPU。
- Web 僅公開 127.0.0.1:8080；API 與模型服務透過 Docker 網路連線。
- 舊的 8765 原生 Python 服務已停止。

## 目前 Pose 權重

已切換至 animal_pose_v2.pt（epoch120.pt 快照）並重新建立 Pose 容器；54 類、17 點與眼點索引 0／1 維持一致。容器內 SHA-256 與來源相符，CPU 分析與下載再次通過，見 [新版模型驗證](POSE_EPOCH120_VERIFICATION.json)。

## 已通過

- 30 項單元與服務整合測試，包含 CPU 強制模式、HTTP 模型回應、遠端 Mask 還原、眼距計算、空偵測與 TTL 清理。
- 基本 Compose 及 NVIDIA override 設定解析；本次只執行基本 CPU 部署，未實測 NVIDIA 硬體。
- 從 Docker Web 入口呼叫 API，使用羊群範例完成實際推論：5 個 Seg 實例、2 個 Pose 配對。
- source.jpg、overlay.jpg、result.json、measurements.csv、animal_0_mask.png 均可下載。
- Swagger、OpenAPI、測試範例與目前前端 HTML 可經 Web 入口存取。
- 已在應用內瀏覽器開啟 Docker 網頁，確認 CPU 就緒、三組橫向範例及設定入口；實際按下開始分析後，畫面顯示 5 隻動物、2 組有效雙眼量測與重新分析按鈕。
- API 輸出位於 512 MiB tmpfs，不掛載永久結果 Volume；完成／失敗後預設 1800 秒到期清除。TTL 的自動清理已由短 TTL 整合測試驗證。
- 本機指標與面試頁未納入 project，Git 與 Docker 都排除該檔。

這些結果驗證部署及流程，不代表眼點或距離的正式準確率評估。

## 建置相容性修正

- 基本映像從官方 CPU 套件來源安裝 PyTorch，避免下載不必要的 CUDA 依賴。
- OpenCV 5 的 ARM wheel 在目前虛擬 CPU 匯入時發生 Illegal instruction；容器固定使用已驗證的 OpenCV 4.11.0.86。

## 重跑

在 project/ 執行：

```bash
docker compose -f docker.compose.yml up -d --build
docker compose -f docker.compose.yml ps
python scripts/verify_docker.py --url http://127.0.0.1:8080
```

使用獨立 Compose 的主機，可將 docker compose 換為 docker-compose。Python 依賴已安裝的環境亦可執行：

```bash
python -m unittest discover -s tests -v
python scripts/verify_http_models.py
```

原生 HTTP 測試暫用 18761／18762，完成後會自行關閉兩個測試程序。
