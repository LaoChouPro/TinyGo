# TinyGo

<div align="center">

[English](README_EN.md) | [简体中文](../README.md) | [日本語](README_JP.md)

</div>

ResNetニューラルネットワークに基づいた超軽量囲碁AIです。KataGoの自己対戦データから学習し、最小限のパラメータで高いレベルの対局能力を実現しました。また、勝率予測機能と、オプションのモンテカルロ木探索（MCTS）拡張モードも提供しています。

TinyGoは、一般消費者向けのハードウェアでスムーズに動作する囲碁AIを作成することを目指しており、同時にResNetニューラルネットワークの構築と学習プロセスを学び理解するための最適なプロジェクトでもあります。

主な特徴は以下の通りです：

- **軽量ResNetアーキテクチャ**: 囲碁に最適化されたカスタム深層残差ネットワーク（ResNet）を採用しています。

- 2つのモデルバージョン：
  *   通常モデル (`/src`)：6 残差ブロック、64 チャンネル。
  *   大型モデル (`/src_large`)：10 残差ブロック、128 チャンネル。
  *   超大型モデル (`/extra_large`)：20 残差ブロック、256 チャンネル。注意：このモデルは大規模な学習を行っていません。

*   **教師あり学習 (Supervised Learning)**：3000万以上の高品質なKataGo自己対戦手を使用して学習し、Policy Networkの直感がトップクラスのAIに近づくようにしています。
*   **勝率予測モジュール (`/win_rate`)**: 独立したValue Networkで、本質的には平均二乗誤差回帰タスクであり、現在の局面の勝率（0-100%）を評価できます。同様にKataGoの自己対戦データを使用して学習しています。
*   **MCTS モンテカルロ木探索**: Policy Network（着手提案）とValue Network（局面評価）を組み合わせた完全な探索アルゴリズムで、中盤の戦いや死活計算能力を大幅に向上させます。
*   Tkinterベースのグラフィカルインターフェースを内蔵し、人間対AIの対局、形勢判断表示、待った機能などをサポートしています。
*   **クロスプラットフォーム対応**: Linux, macOS (Apple Silicon MPS アクセラレーション対応), Windows (NVIDIA CUDA アクセラレーション対応) で動作します。

## 成果

超大規模な学習を行っていない状態で：

Largeモデルはデータの4%（120万手サンプル）を使用して142エポック学習し、Top1正解率は38.7%、Top5正解率は70.4%でした。

Largeモデル + MCTSx100は、KataGoの「直感（レベル1）」モードのレベルを打ち負かし、改善の余地が非常に大きいです。

![KataGo-winrate](../imgs/KataGo-winrate.png)

*画像中の黒番はTinyGo-large-MCTSx100、白番はKataGo「直感（レベル1）」モードです。*

MCTSを使用せず、Largeモデルのみで推論を行った場合、RTX 5070 Ti GPU上での1手あたりの推論速度は驚異の0.04秒に達し、KataGoの「直感（レベル1）」モードのレベルとほぼ同等です。

`win_rate`のSmallモデルは、400万サンプルを使用して18エポック学習した後、MSEは0.042、MAEは0.112に達し、MCTS分析に基本的に使用可能です。

### データ表現

モデル入力は `(Batch_Size, 3, 19, 19)` のテンソルで、3つの特徴平面を含みます：

1.  **自分の石**: 現在の手番の石の位置（石があれば1、なければ0）。
2.  **相手の石**: 相手の石の位置。
3.  **バイアス平面 (Bias)**: 全て1のマトリックスで、ネットワークが空点情報や境界特徴を捉えるのを助けるために使用されます。

### ネットワークアーキテクチャ

TinyGoは、古典的なAlphaZeroスタイルのResNetアーキテクチャを採用しています：

*   **初期ブロック (Initial Block)**: 3x3 畳み込み核、Stride=1、Padding=1、Batch Normalization と ReLU 活性化が続きます。
*   **残差タワー (Residual Tower)**: $N$ 個の残差ブロックを積み重ねます。
    *   各残差ブロックの内容：`Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add Skip Connection -> ReLU`。
*   **ポリシーヘッド (Policy Head)**:
    *   1x1 畳み込み、出力チャンネル数 2。
    *   Batch Normalization + ReLU。
    *   全結合層 (Linear): 入力 `2 * 19 * 19`、出力 `361` (19x19の着手点に対応)。
*   **バリューヘッド (Value Head)** (勝率モデルのみ):
    *   1x1 畳み込み、出力チャンネル数 1。
    *   全結合層 -> ReLU -> 全結合層 -> Sigmoid、0〜1の間のスカラーを出力。

### 学習設定

*   **オプティマイザ**: Adam
*   **学習率スケジューリング**: `ReduceLROnPlateau` (検証セットの精度が向上しなくなった場合、学習率を0.5倍に減衰)。
*   **データ拡張 (Data Augmentation)**: データを最大限に活用するため、学習中にGPU上でリアルタイムに8種類の対称変換（回転 0/90/180/270度 $\times$ 反転の有無）を行います。
*   **損失関数**: CrossEntropyLoss (Policy Network), MSE/BCE (Value Network)。

### MCTS アルゴリズム

*   **PUCT 公式**: 「活用」(Exploitation, 勝率の高い点を選ぶ) と 「探索」(Exploration, あまり訪問していない点を選ぶ) のバランスをとるために使用されます。
    *   $U(s, a) = C_{puct} \times P(s, a) \times \frac{\sqrt{N(s)}}{1 + N(s, a)}$
*   **シミュレーション回数**: デフォルト 100 回/手 (設定可能)。
*   **マルチスレッド**: Pythonレベルでのバッチ評価による探索の高速化をサポートしています。

## インストールガイド

1. **プロジェクトコードのクローン**

   ```bash
   git clone https://github.com/LaoChouPro/TinyGo.git
   cd TinyGo
   ```

2. **依存関係のインストール**
   Pythonのバージョンが $\ge$ 3.8 であることを確認してください。

   ```bash
   pip install -r requirements.txt
   ```

   *   **GPU アクセラレーションのヒント**:
       *   **NVIDIA ユーザー**: CUDA バージョンの PyTorch をインストールしてください (例: `pip install torch --index-url https://download.pytorch.org/whl/cu118`)。
       *   **Mac ユーザー**: PyTorch はデフォルトで MPS (Metal Performance Shaders) をサポートしているため、追加の操作は不要です。

## TinyGoを使用した対局と継続学習

### 1. 人間対AI対局 (Play)

**標準モード (高速応答)**

Policy Networkのみを使用して着手し、速度が非常に速く、布石や定石の迅速なテストに適しています。

注意：このスクリプトはLargeモデルを使用しています。ソースコード内でモデルパスを変更できます。

```bash
python play_gui.py
```

**MCTS 拡張モード (最強棋力)**
モンテカルロ木探索を有効にし、勝率評価と組み合わせて、深い計算能力を備えています。

```bash
python play_gui_mcts.py
```

### 2. モデル学習 (Train)

既存のデータを使用して学習を継続したい場合（現在のモデルは超大規模な学習を行っていません）、または学習結果を再現したい場合は、以下のスクリプトを使用できます。

**Policy Network の学習**

```bash
python src/train.py # 同様に、python src_large/train.py を使用してLargeモデルを学習できます。
```

対話式プロンプトに従って設定してください：

*   `Epochs`: 学習エポック数。注意：この数は以前学習したエポック数から継続する必要があります。例えば、bestモデルが30エポックまで学習されている場合、35エポックと入力して5エポック継続学習させます。
*   `Batch Size`: バッチサイズ (推奨 64-1024, VRAMによる)。
*   `Learning Rate`: 学習率 (デフォルト 0.001, オプティマイザが自動的にスケジューリングします)。

**Value Network の学習**

```bash
cd win_rate
python process_data.py  # ステップ1：データ前処理
python src_small/train.py  # ステップ2：学習開始
```

## プロジェクト構造の詳細

*   `src/`: **標準モデルソース (Standard Model)**
    *   `model.py`: TinyGoNet (6 blocks) 構造を定義。
    *   `train.py`: メインの学習ループ。データロードと検証ロジックを含む。
    *   `dataset.py`: SGFデータを処理し、特徴テンソルを生成。
*   `src_large/`: **大型モデルソース (Large Model)**
    *   10 blocks, 128 channels のモデル定義を含む。
*   `src_extra_large/`: **超大型モデルソース (Extra Large Model)**
    *   20 blocks, 256 channels のモデル定義を含む。
*   `src_mcts/`: **モンテカルロ木探索コア**
    *   `mcts.py`: MCTS メインループのロジック。
    *   `node.py`: ツリーノードの定義とPUCT選択式。
*   `win_rate/`: **勝率予測モジュール**
    *   独立したデータ処理とValue Network学習コードを含む。
*   `play_gui.py`: 標準モード起動エントリ。
*   `play_gui_mcts.py`: MCTSモード起動エントリ。

## 謝辞

学習データは、オープンソースの **KataGo** 自己対戦棋譜 (https://katagoarchive.org) から提供されています。
