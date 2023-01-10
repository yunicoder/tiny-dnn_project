## tiny-dnn_project

### Usage
1. ソースコードのclone
`git clone https://github.com/tiny-dnn/tiny-dnn.git`

2. tiny-dnnをincludeパスに追加

[プロジェクト]>[プロパティ]>[C/C++]>[全般]の中にある[追加のインクルードディレクトリ]にcloneしてきたパスを書く

3. パスの変更

MNISTの入っているデータパスを変更
```
std::string data_path = "C:\\Users\\tkmpa\\workdir\\Github\\tiny-dnn\\data";  // datapathを自分で指定
```

4. ビルド&実行
```
$ cd source
$ make
```

