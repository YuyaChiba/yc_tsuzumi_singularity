## ディレクトリ構造
- data
  - 学習データ保存用のディレクトリ: 中に学習データ (train.jsonl)、評価データ (valid.jsonl)が存在することを想定
- experiments
  - 学習結果出力先のディレクトリ
- finetune_scripts
  - 学習スクリプトのディレクトリ
- llm-foundry
  - tsuzumi用にカスタマイズされたllm-foundryをこの階層に設置

## ビルド
### 注意点
- ビルドはホーム領域かスクラッチ領域を推奨
  - ノードはどこでも問題ありません
- sandboxでビルドするので、実体がこのディレクトリに保存されます
- 実行する際にはビルドで作成されたntt-llm-tools-20240216.sifが直下に必要です
- full fine-tuningをする予定がある場合は先に「Full fine-tuningをする場合」もご確認ください
### 実施方法
1. 事前準備  
   - tsuzumiのllm_foundryを直下にコピー
   - launch_singularity.sh内のMODEL_SOURCE_DIRを環境に合わせて設定
     - 直下にtsuzumi (v1_02-7b-instruct-hf)があるディレクトリを想定しています
3. singularityのモジュールをロード
   ~~~
   module load singularitypro
   ~~~
4. コンテナをビルド
   ~~~
   ./build_singularity.sh
   ~~~
5. 実行
   ~~~
   ./launch_singularity.sh
   ~~~
### Full fine-tuningをする場合
- DeepSpeed Zero3を使いますので、deepspeedのインストールが必要です
- インストール方法
  - singularityを実行し、pip3でインストール
    ~~~
    ./launch_singularity.sh
    pip install deepspeed # PythonのPathがおかしい場合は、pip3.10など、使うPythonを明示するのがベター
    ~~~
  - または、定義ファイル (tsuzumi.def)に直接追記してビルド
    - こちら未検証です
### ビルドファイルの作成方法
- tsuzumi.defはsingularity-pythonを使ってDockerFileを変換して作成
- launch_singularity.sh, build_singularity.shはgpt4-turboを使ってlauch.sh, build.shを変換して作成

## Full fine-tuningの実施
### 注意点
- LoRAチューニングを行う場合
  - 配布されているスクリプトがLoRAチューニングを想定しているのでそちらをお使いいただくのがベターです
- 学習・推論スクリプト実行時にモジュールが適切にロードされない場合は踏み台にしているPython (= 3.10)が正しく使われているか確認

### 実施方法
1. 事前準備
   - Aノードを8GPUを確保 (= rt_AF=1)
   - データのフォーマットなどは想定通り読まれているか確認してください
     - CS研の実験で用いているデータに準拠していますので、多少lora.pyの改造が必要だと思います
2. インタラクティブノードでの実行
   ~~~
   module load singularitypro
   ./launch_singularity.sh
   /finetune_scripts/run/run_multi_full_abci.sh
   ~~~
3. バッチジョブの実行
   ~~~
   qsub -g <group_id> batchjob_singularity.sh
   ~~~
