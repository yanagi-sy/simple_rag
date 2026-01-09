# 単一ファイル対応RAGシステム

## 概要

単一のPDFファイルまたはテキストファイルを読み込んで、対話型の質問応答システムを構築するRAG（Retrieval-Augmented Generation）システムです。

ハイブリッド検索（セマンティック検索とキーワード検索）とリランキングを使用して、精度の高い回答を生成します。

### 主な機能

- 単一ファイル対応（PDFまたはテキストファイル）
- ハイブリッド検索（Semantic + Keyword）
- リランキング（CrossEncoder）
- ターミナル風UI（Tkinter）
- チャンク設定：chunk_size=1200, chunk_overlap=200

## 技術要件

### 必要な環境

- **Python**: 3.11以上
- **Ollama**: LLM実行環境（llama3:latestモデルが必要）
- **Tkinter**: UI用（通常はPythonに同梱）

### 必要なパッケージ

```
langchain
langchain-community
langchain-text-splitters
sentence-transformers
chromadb
ollama
```

## 環境構築手順

### 1. 仮想環境の作成とアクティベート

```bash
# 仮想環境を作成
python3 -m venv venv311

# 仮想環境をアクティベート
source venv311/bin/activate  # macOS/Linux
# または
venv311\Scripts\activate  # Windows
```

### 2. 必要なパッケージのインストール

```bash
pip install langchain langchain-community langchain-text-splitters sentence-transformers chromadb ollama
```

### 3. Ollamaのセットアップ

```bash
# Ollamaを起動（別のターミナルで実行）
ollama serve

# llama3モデルをインストール
ollama pull llama3:latest
```

### 4. アプリケーションの実行

```bash
# プロジェクトディレクトリに移動
cd ~/simple_rag.py

# 実行スクリプトを使用（推奨）
./scripts/run_ui.sh

# または、手動で実行
source venv311/bin/activate
python -m src.ui.terminal
```

## ディレクトリ構成

```
simple_rag.py/
├── src/                               # ソースコード
│   ├── __init__.py
│   ├── rag/                           # RAGシステムコア
│   │   ├── __init__.py
│   │   ├── loaders/                   # データローダー
│   │   │   ├── __init__.py
│   │   │   └── pdf.py                 # PDF/テキスト読み込み処理（PDFRagSystem）
│   │   ├── models/                    # RAGモデル
│   │   │   ├── __init__.py
│   │   │   └── single_source.py       # 単一ソース対応ReRankingRAG
│   │   └── utils/                     # ユーティリティ関数
│   │       ├── __init__.py
│   │       └── text.py                # テキストクリーニング関数
│   └── ui/                            # ユーザーインターフェース
│       ├── __init__.py
│       ├── terminal.py                # メインクラス（RAGTerminalUI）
│       ├── ui_builder.py              # UI構築
│       ├── message_handler.py        # メッセージ表示
│       ├── event_handler.py          # イベントハンドラー
│       ├── source_manager.py         # ソース管理
│       └── rag_integration.py        # RAG統合
├── scripts/                           # 実行スクリプト
│   └── run_ui.sh                      # UI起動スクリプト
├── chroma_db/                         # ベクトルデータベース（自動生成）
├── README.md                          # このファイル
└── venv311/                           # 仮想環境
```

### 主要クラス

- **PDFRagSystem** (`src/rag/loaders/pdf.py`): PDFまたはテキストファイルを読み込んでチャンク化
- **ReRankingRAG** (`src/rag/models/single_source.py`): ハイブリッド検索とリランキングによる回答生成
- **RAGTerminalUI** (`src/ui/terminal.py`): メインUIクラス
