# 四容器系統架構

```mermaid
flowchart LR
    USER[瀏覽器] -->|HTTP 對外入口| WEB
    subgraph NETWORK[Docker Compose 內部網路]
        WEB[web 容器：靜態網頁與反向代理] -->|同源 /api| API[api 容器：工作狀態、配對與量測]
        API -->|內部 HTTP| SEG[seg 容器：YOLO26n-seg]
        API -->|內部 HTTP| POSE[pose 容器：YOLO26n-pose]
        SEG --- SW[唯讀 Seg 權重]
        POSE --- PW[唯讀 Pose 權重]
        API --> TMP[當次影像及結果暫存：到期清除]
    end
    API -->|分析狀態與下載結果，經 Web 代理| USER
```

1. Web 提供頁面並將 `/api` 請求轉送 API。
2. API 接收圖片，管理當次工作並呼叫 Seg、Pose。
3. Seg 回傳動物框與遮罩，Pose 回傳物種、框及 17 個關鍵點。
4. API 執行物種與 IoU 配對、左右眼檢查、雙眼及跨動物右眼距離計算。
5. 網頁顯示結果，提供 JSON／CSV／標示圖下載，暫存到期清除。

共四個容器；不建立歷史列表、歷史資料庫或永久結果 Volume。暫存可採 API 容器 tmpfs，須限制容量並實作到期清理。服務重啟可清除當次狀態。

模型服務就緒後啟動 API，API 就緒後由 Web 提供入口。模型權重採唯讀掛載，模型服務與 API 不直接公開連接埠。NVIDIA GPU 需另配置裝置資源；無 GPU 使用 CPU。

已提供四服務 Compose 與 Dockerfile；原生入口仍保留單程序推論方式。部署驗證見 DOCKER_VERIFICATION.md。
