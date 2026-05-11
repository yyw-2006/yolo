# 图片样例目录说明

该目录用于放置本机/虚拟机测试图片，目录名应尽量与 `config/category_mapping.json` 中的类别名一致，例如：

- `samples/normal/`
- `samples/politics/`
- `samples/protest/`
- `samples/fallen_official/`
- `samples/disgraced_artist/`
- `samples/terrorism/`
- `samples/drugs/`
- `samples/illegal_religion/`
- `samples/gambling/`
- `samples/nazi_symbol/`
- `samples/obscene_gesture/`

公开样本导入示例：

```bash
SOURCE=github_hate_symbols CATEGORY=nazi_symbol TAKE=2 python3 -m tests.import_public_samples
```

如果公开源需要账号、token、授权或人工申请，脚本会输出获取说明，不会把凭证写入代码。

接口调用示例：

```bash
curl -X POST "http://127.0.0.1:8000/audit/image" \
  -F "file=@samples/nazi_symbol/nazi_symbol_Peckerwood_(hand_sign)_3.webp"
```

