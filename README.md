# Singularityメモ
## 注意点
- ビルドはホーム領域かスクラッチ領域を推奨
- lora.py内でデータのキャッシュを保存するため読み書き可能なディレクトリが必要
  - デフォルトはtuzumi_dialogue/.cacheなので、ない場合は作成してください
- 学習・推論スクリプト実行時にモジュールが適切にロードされない場合は踏み台にしているPython (= 3.10)が正しく使われているか確認
## ABCIでのビルド
tsuzumi_dialogue直下で下記を実行
1. singularityのモジュールをロード
   ~~~
   module load singularitypro
   ~~~
2. コンテナをビルド
   ~~~
   ./build_singularity.sh
   ~~~
3. 実行
   ~~~
   ./launch_singularity.sh
   ~~~
## ビルドファイルの作成方法
- tsuzumi.defはsingularity-pythonを使ってDockerFileを変換して作成
  - 水上さんのDockerFileに合わせて編集済み
- launch_singularity.sh, build_singularity.shはgpt4-turboを使ってlauch.sh, build.shを変換して作成
  - 実験用に編集済み
