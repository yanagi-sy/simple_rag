# 複数ソース対応RAGシステムの処理フロー解説

## 概要

このドキュメントは、複数ソース（PDF/テキストファイル/手動テキスト）に対応したRAGシステムの処理フローを、初心者向けに詳しく説明します。

## システム全体の構成

### 1. クラス構成

```
pdf_rag.py
├── PDFRagSystem（既存）: 単一PDFファイルを読み込むクラス
├── ReRankingRAG（既存）: 検索と回答生成を行うクラス
├── MultiSourceRagSystem（新規）: 複数ソースを統合管理するクラス
└── MultiSourceReRankingRAG（新規）: 複数ソース対応の検索・回答生成クラス

rag_terminal_ui.py
└── RAGTerminalUI（拡張）: UIとRAGシステムを統合するクラス
```

### 2. データフロー

```
ユーザー入力
    ↓
[UI] ソース追加（PDF/テキストファイル/手動テキスト）
    ↓
[MultiSourceRagSystem] チャンク化・メタデータ付与
    ↓
[MultiSourceRagSystem] インデックス構築（BM25 + ベクトルDB）
    ↓
[UI] ユーザーが質問を入力
    ↓
[MultiSourceReRankingRAG] ハイブリッド検索（Semantic + Keyword）
    ↓
[MultiSourceReRankingRAG] リランキング（CrossEncoder）
    ↓
[MultiSourceReRankingRAG] LLMで回答生成
    ↓
[UI] 回答を表示
```

## 詳細な処理フロー

### フェーズ1: ソースの追加

#### 1-1. PDFファイルの追加

**処理の流れ：**

1. **ユーザー操作**
   - 「複数PDF選択」ボタンをクリック
   - ファイル選択ダイアログで複数のPDFファイルを選択

2. **UI処理（`_select_multiple_pdfs`メソッド）**
   - `filedialog.askopenfilenames()`で複数ファイルを選択
   - 選択された各PDFファイルに対して`_add_pdf_to_system()`を呼び出す

3. **PDF追加処理（`_add_pdf_to_system`メソッド）**
   - `MultiSourceRagSystem`が未初期化の場合は初期化
   - `multi_source_system.add_pdf(pdf_path)`を呼び出す

4. **PDF読み込み処理（`MultiSourceRagSystem.add_pdf`メソッド）**
   - `PyPDFLoader`でPDFを読み込む（ページごとに分割）
   - `RecursiveCharacterTextSplitter`でチャンクに分割
   - 各チャンクに以下のmetadataを付与：
     - `source_type`: "pdf"
     - `source_name`: ファイル名
     - `chunk_id`: チャンク識別子
   - チャンクを`self.docs`リストに追加
   - ソース情報を`self.sources`リストに記録

5. **UI更新**
   - ソース一覧を更新（`_update_source_list`）
   - 成功メッセージを表示

#### 1-2. テキストファイルの追加

**処理の流れ：**

1. **ユーザー操作**
   - 「テキストファイル追加」ボタンをクリック
   - ファイル選択ダイアログでテキストファイル（.txt）を選択

2. **テキストファイル読み込み処理（`MultiSourceRagSystem.add_text_file`メソッド）**
   - ファイルをUTF-8で読み込む（失敗した場合はShift_JISを試す）
   - `Document`オブジェクトに変換
   - `RecursiveCharacterTextSplitter`でチャンクに分割
   - 各チャンクに以下のmetadataを付与：
     - `source_type`: "text_file"
     - `source_name`: ファイル名
     - `chunk_id`: チャンク識別子
   - チャンクを`self.docs`リストに追加
   - ソース情報を`self.sources`リストに記録

#### 1-3. 手動テキストの追加

**処理の流れ：**

1. **ユーザー操作**
   - 「テキスト直接入力」ボタンをクリック
   - ダイアログが開く

2. **テキスト入力ダイアログ（`_open_manual_text_dialog`メソッド）**
   - 新しいウィンドウ（`Toplevel`）を開く
   - ソース名入力フィールドとテキスト入力エリアを表示
   - ユーザーがテキストを入力して「追加」ボタンをクリック

3. **手動テキスト追加処理（`MultiSourceRagSystem.add_manual_text`メソッド）**
   - 入力されたテキストを`Document`オブジェクトに変換
   - `RecursiveCharacterTextSplitter`でチャンクに分割
   - 各チャンクに以下のmetadataを付与：
     - `source_type`: "manual_text"
     - `source_name`: ユーザーが指定した識別名
     - `chunk_id`: チャンク識別子
   - チャンクを`self.docs`リストに追加
   - ソース情報を`self.sources`リストに記録

### フェーズ2: インデックス構築

#### 2-1. インデックス構築の開始

**処理の流れ：**

1. **ユーザー操作**
   - すべてのソースを追加した後、「インデックス構築」ボタンをクリック

2. **インデックス構築処理（`_build_index`メソッド）**
   - `MultiSourceRagSystem`が初期化されているか確認
   - チャンクが存在するか確認
   - 別スレッドでインデックス構築を開始（UIがフリーズしないように）

3. **インデックス構築の実行（`MultiSourceRagSystem.build_index`メソッド）**
   - 既存のChroma DBを削除（古いデータをクリア）
   - **BM25インデックスの構築**
     - すべてのチャンクを統合して1つのBM25インデックスを作成
     - これにより、キーワード検索が可能になる
   - **ベクトルDBの構築**
     - すべてのチャンクを384次元のベクトルに変換
     - Chroma DBに保存
     - これにより、セマンティック検索が可能になる

4. **RAGシステムの初期化**
   - `MultiSourceReRankingRAG`を初期化
   - 構築済みのBM25とベクトルDBを使用
   - プロンプトテンプレートを設定

5. **UI更新**
   - 構築完了メッセージを表示
   - ソース数と総チャンク数を表示

### フェーズ3: 質問応答

#### 3-1. 質問の送信

**処理の流れ：**

1. **ユーザー操作**
   - 質問を入力フィールドに入力
   - 「送信」ボタンをクリック、またはEnterキーを押す

2. **質問送信処理（`_send_message`メソッド）**
   - 入力フィールドから質問を取得
   - ユーザーメッセージをチャット表示エリアに表示
   - RAGシステムが初期化されているか確認
   - 別スレッドで検索と回答生成を開始

#### 3-2. 検索の実行

**処理の流れ：**

1. **検索重みの取得**
   - UIで設定されたSemantic検索の重み（w_sem）を取得
   - UIで設定されたKeyword検索の重み（w_key）を取得

2. **ハイブリッド検索（`MultiSourceReRankingRAG.search`メソッド）**
   - **セマンティック検索**
     - 質問を384次元のベクトルに変換
     - ベクトルDBから類似度の高いチャンクを検索（上位60件）
   - **キーワード検索（BM25）**
     - 質問に含まれる単語が多く出現するチャンクを検索（上位60件）
   - **統合**
     - `EnsembleRetriever`で2つの検索結果を統合
     - 重み（w_sem, w_key）に応じて結果を調整

3. **リランキング（`MultiSourceReRankingRAG.search`メソッド内）**
   - **CrossEncoderによる再評価**
     - 質問と各候補チャンクのペアを作成
     - CrossEncoderで関連度スコアを計算（0.0〜1.0）
   - **並び替え**
     - スコアの高い順に並び替え
     - 上位5件を選択

4. **ソース情報の取得**
   - 各チャンクのmetadataから`source_type`と`source_name`を取得
   - どのソースから情報が得られたかを集計

#### 3-3. 回答の生成

**処理の流れ：**

1. **文脈の構築**
   - 検索結果の各チャンクの内容を取得
   - ソース情報を付けて結合
   - 例：「[PDF: linuxtext_ver4.0.0.pdf]\nチャンクの内容...」

2. **プロンプトの生成**
   - プロンプトテンプレートに以下を埋め込む：
     - `context`: 検索結果（ソース情報付き）
     - `question`: ユーザーの質問

3. **LLMによる回答生成**
   - Ollamaのllama3モデルにプロンプトを渡す
   - LLMが回答を生成
   - 回答をチャット表示エリアに表示

4. **会話履歴の記録**
   - ユーザーの質問とアシスタントの回答を会話履歴に追加

## 重要な設計ポイント

### 1. 既存コードの維持

- `PDFRagSystem`と`ReRankingRAG`は変更せず、そのまま使用可能
- 新しいクラス（`MultiSourceRagSystem`、`MultiSourceReRankingRAG`）を追加
- 既存のコードを壊さない設計

### 2. メタデータの付与

各チャンクに必ず以下のメタデータを付与：
- `source_type`: ソースの種類（"pdf" / "text_file" / "manual_text"）
- `source_name`: ソース名（ファイル名 or 識別名）
- `chunk_id`: チャンク識別子

これにより、検索結果からどのソース由来の情報かが分かる。

### 3. 統合インデックス

- すべてのソースを1つの知識ベースとして扱う
- BM25とベクトルDBは、すべてのチャンクを統合して構築
- 横断検索が可能（NotebookLMのような動作）

### 4. 検索と生成の分離

- 検索：`search()`メソッド（ソース情報を含むDocumentリストを返す）
- 生成：`answer()`メソッド（LLMを呼び出して回答を生成）
- LLMは最後に1回だけ呼ぶ（設計方針を遵守）

## 使用例

### 基本的な使用フロー

1. **ソースの追加**
   ```
   1. 「複数PDF選択」でPDFファイルを追加
   2. 「テキストファイル追加」でテキストファイルを追加
   3. 「テキスト直接入力」で手動テキストを追加
   ```

2. **インデックス構築**
   ```
   「インデックス構築」ボタンをクリック
   → すべてのソースを統合して検索可能なインデックスを構築
   ```

3. **質問応答**
   ```
   質問を入力 → 送信
   → すべてのソースから横断的に検索
   → LLMが回答を生成
   ```

### 検索結果の例

検索結果には、各チャンクのソース情報が含まれます：

```
[PDF: linuxtext_ver4.0.0.pdf]
systemctlコマンドは、systemdシステムのサービスを管理するための...

[TXT: notes.txt]
サービス管理の基本コマンドは以下の通りです...

[手動: メモ]
重要なポイント：systemctl start でサービスを起動...
```

これにより、どのソースから情報が得られたかが一目で分かります。

## まとめ

このシステムは、既存のRAGシステムの設計思想を維持しつつ、複数のソース（PDF/テキストファイル/手動テキスト）を統合して、NotebookLMのような横断検索を実現します。各チャンクにソース情報を付与することで、検索結果の出所が明確になり、信頼性の高い質問応答システムを構築できます。

