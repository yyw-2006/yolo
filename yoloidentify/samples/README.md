图片样例目录说明

该目录用于放置你本机/虚拟机的测试图片（不随代码仓库提供二进制图片）。

建议你自行准备并命名，例如：
- `samples/normal.jpg`
- `samples/violence.jpg`
- `samples/flag.jpg`
- `samples/person.jpg`

调用示例：
```bash
curl -X POST "http://127.0.0.1:8000/audit/image" \
  -F "file=@samples/normal.jpg"
```

