# 四容器 Docker 部署

部署步驟、環境設定與指令以 [README 第 3 節](../README.md#3-部署方式dockerfile-與-dockercomposeyml) 為準。

- 主配置：`../docker.compose.yml`。
- NVIDIA 配置：`../docker.compose.gpu.yml`（需實際 GPU 主機驗證）。
- Dockerfile：`../docker/web.Dockerfile`、`../docker/api.Dockerfile`、`../docker/model.Dockerfile`。
- Nginx 代理：`../docker/nginx.conf.template`。
- 模型唯讀掛載，只有 Web 公開連接埠，無歷史資料庫。
- API 當次結果使用 512 MiB tmpfs，完成／失敗後預設 1800 秒清除。
- 一次僅執行一筆工作，模型 HTTP 請求 timeout 180 秒；尚無取消 API。

實際建置／推論驗證結果見 [DOCKER_VERIFICATION.md](DOCKER_VERIFICATION.md)。
