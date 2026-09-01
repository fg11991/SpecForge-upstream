# CLAUDE.md

## Git 约定

### 提交身份

所有提交都用这个身份，不要用容器默认的 `Claude <noreply@anthropic.com>`：

```bash
git config user.name "maplesong"
git config user.email "fgwuyidong@gmail.com"
```

每个新 session 开始工作前先设一次（容器是临时的，上一次的设置不会保留）。

### 提交信息

**不要**在提交信息末尾附加 `Co-Authored-By:` 或 `Claude-Session:` 尾注。
提交信息只写改动本身。

### 推送

推送到指定的开发分支，用 `git push -u origin <branch-name>`。
不要在未经明确要求时创建 pull request。
