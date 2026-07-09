# 版本管理与更新说明

这份文档用于维护 `index-tts-studio` 这个派生项目。

## 分支策略

- `main`：稳定可用版本，适合直接部署。
- `feature/*`：新功能开发分支。
- `fix/*`：问题修复分支。
- `upstream-sync/*`：同步官方 IndexTTS 更新时使用。

小型个人项目可以直接在 `main` 开发，但每次推送前至少运行语法检查和一次 WebUI 启动验证。

## 版本号规则

建议使用语义化版本：

- `0.1.1`：当前 Studio 初版维护版本。
- `0.1.0`：Studio 初版。
- `0.1.x`：小修复，例如 UI 可读性、错误提示、部署配置。
- `0.2.0`：新增用户可感知功能，例如音色库元数据、批量任务。
- `1.0.0`：形成稳定产品形态，安装、配置、生成、保存、部署流程都有文档和验证。

## 每次更新要做什么

1. 修改代码。
2. 更新 `CHANGELOG.md`。
3. 如果使用方式变化，更新 `README.md`。
4. 如果涉及部署，更新本文件或新增部署文档。
5. 运行检查：

```bash
uv run python -m py_compile webui.py
```

6. 启动 WebUI，至少验证：

- 页面可打开。
- 本地音频上传后能处理为参考音色。
- 文本能生成音频。
- 如果改动了网络素材，验证 `yt-dlp` 仍可解析一个公开素材。

## 同步官方上游

当前项目来自官方仓库：

```bash
git remote add upstream https://github.com/index-tts/index-tts.git
git fetch upstream
```

同步时建议开新分支：

```bash
git checkout -b upstream-sync/2026-xx-xx
git merge upstream/main
```

合并后重点检查：

- `webui.py` 是否有官方更新与 Studio 改造冲突。
- `pyproject.toml` 和 `uv.lock` 的依赖变化。
- `docs/README_zh.md` 是否需要同步官方说明。
- 模型许可和下载方式是否变化。

## 发版建议

打 tag：

```bash
git tag -a v0.1.1 -m "IndexTTS Studio 0.1.1"
git push origin v0.1.1
```

GitHub Release 描述建议包含：

- 新增功能
- 修复问题
- 升级注意事项
- 是否需要重新下载模型或重新安装依赖
