"""
RAG Terminal UI - Tkinterを用いたターミナル風チャットUI

既存のRAGシステム（pdf_rag.py）を利用して、
ターミナル風のチャットインターフェースを提供します。

このプログラムの主な機能：
1. PDFファイルを読み込んで検索可能な形式に変換
2. ユーザーの質問に対して、PDFから関連情報を検索
3. LLM（大規模言語モデル）を使って質問に回答
4. ターミナル風のUIで対話的に質問応答が可能

用語説明：
- RAG: Retrieval-Augmented Generation（検索拡張生成）
  PDFから情報を検索して、その情報を元にLLMが回答を生成する技術
- Tkinter: Pythonの標準GUIライブラリ
- LLM: 大規模言語モデル（このプログラムではOllamaのllama3を使用）
"""

# =========================================
# ライブラリのインポート
# =========================================
# Tkinter: GUIアプリケーションを作成するための標準ライブラリ
import tkinter as tk
# scrolledtext: スクロール可能なテキストエリアを作成
from tkinter import scrolledtext, messagebox, filedialog
# ttk: Tkinterの拡張ウィジェット（今回は使用していませんが、将来の拡張用）
from tkinter import ttk
# threading: 別スレッドで処理を実行するためのライブラリ
# （重い処理を別スレッドで実行することで、UIがフリーズしないようにする）
import threading
# datetime: 日時を扱うためのライブラリ（タイムスタンプ表示用）
from datetime import datetime
# typing: 型ヒント（変数の型を明示するための機能）
from typing import Optional
# os: オペレーティングシステム関連の機能（ファイルパス操作など）
import os
# sys: システム関連の機能（今回は使用していませんが、将来の拡張用）
import sys

# =========================================
# 既存のRAGコードをインポート
# =========================================
# pdf_rag.pyから、PDFを読み込むクラスとRAG検索を行うクラスをインポート
from pdf_rag import PDFRagSystem, ReRankingRAG, MultiSourceRagSystem, MultiSourceReRankingRAG
# PromptTemplate: LLMへの指示文を管理するためのテンプレート
from langchain.prompts import PromptTemplate
# Document: 文書データを表現するためのクラス
from langchain.docstore.document import Document


class RAGTerminalUI:
    """
    Tkinterを用いたターミナル風チャットUI
    
    既存のRAGシステムと統合して、対話型の質問応答システムを提供します。
    """
    
    def __init__(self, root: tk.Tk):
        """
        初期化メソッド
        
        このメソッドは、アプリケーションを起動する際に最初に呼ばれます。
        UIの設定や、RAGシステムの準備を行います。
        
        Args:
            root: Tkinterのルートウィンドウ（メインウィンドウ）
        """
        # =========================================
        # ウィンドウの基本設定
        # =========================================
        self.root = root
        # ウィンドウのタイトルを設定
        self.root.title("RAG Terminal UI - 複数ソース対応質問応答システム")
        # ウィンドウのサイズを設定（幅1000px、高さ700px）
        self.root.geometry("1000x700")
        # ウィンドウの背景色を設定（ダークテーマ: #1e1e1e）
        self.root.configure(bg="#1e1e1e")
        
        # =========================================
        # RAGシステムのインスタンス変数
        # =========================================
        # MultiSourceRagSystem: 複数ソース（PDF/テキストファイル/手動テキスト）を統合管理するクラス
        # 初期状態ではNone（ソースが読み込まれるまで使用不可）
        self.multi_source_system: Optional[MultiSourceRagSystem] = None
        # MultiSourceReRankingRAG: 複数ソース対応の検索と回答生成を行うクラス
        # 初期状態ではNone（ソースが読み込まれるまで使用不可）
        self.rag_system: Optional[MultiSourceReRankingRAG] = None
        
        # 後方互換性のため、既存の変数名も保持（非推奨）
        self.pdf_rag_system: Optional[PDFRagSystem] = None
        
        # =========================================
        # プロンプトテンプレートの設定
        # =========================================
        # プロンプトテンプレート = LLMへの指示文のテンプレート
        # {context} と {question} の部分が後で実際の値に置き換えられる
        # このテンプレートにより、LLMがPDFの内容に基づいて回答を生成する
        self.prompt_template = PromptTemplate(
            template="""あなたは日本語の質問応答システムです。以下の文脈から質問に答えてください。
【文脈】
{context}
【質問】
{question}
【回答の際の注意点】
・文脈に書かれている事実のみを使用する
・推測や一般知識を混ぜない
・答えられない場合は正直にその旨を伝える
・丁寧で分かりやすい日本語で回答する
【回答】""",
            input_variables=["context", "question"]  # テンプレート内で使用する変数名
        )
        
        # =========================================
        # 会話履歴の管理
        # =========================================
        # ユーザーとアシスタントの会話を記録するリスト
        # 各会話は辞書形式で保存される（role, message, timestamp）
        self.conversation_history = []
        
        # =========================================
        # UIの構築と初期化
        # =========================================
        # UIを構築するメソッドを呼び出す
        self._build_ui()
        
        # 初期メッセージを表示（ユーザーへの案内）
        self._add_system_message(
            "RAG Terminal UI にようこそ！\n"
            "複数のPDFファイル、テキストファイル、または手動入力テキストを追加できます。\n"
            "すべてのソースを追加した後、「インデックス構築」ボタンを押してから質問を開始してください。"
        )
    
    def _build_ui(self):
        """
        UIを構築するメソッド
        
        このメソッドは、アプリケーションの画面（UI）を作成します。
        Tkinterのウィジェット（ボタン、テキストエリアなど）を配置して、
        ユーザーが操作できるインターフェースを構築します。
        """
        # =========================================
        # メインフレーム（全体のコンテナ）
        # =========================================
        # Frame = ウィジェットを配置するためのコンテナ
        # fill=tk.BOTH, expand=True = ウィンドウのサイズに合わせて拡張
        # padx, pady = 外側の余白（10ピクセル）
        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # =========================================
        # 上部：ファイル選択と設定エリア
        # =========================================
        # 上部にPDFファイル選択ボタンと検索重みの設定を配置
        top_frame = tk.Frame(main_frame, bg="#1e1e1e")
        top_frame.pack(fill=tk.X, pady=(0, 10))  # fill=tk.X = 横方向に拡張
        
        # =========================================
        # ソース追加ボタン群
        # =========================================
        # 複数のソース（PDF/テキストファイル/手動テキスト）を追加できるボタン群
        button_frame = tk.Frame(top_frame, bg="#1e1e1e")
        button_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        # 複数PDFファイル選択ボタン（新機能）
        multi_pdf_button = tk.Button(
            button_frame,
            text="複数PDF選択",
            command=self._select_multiple_pdfs,
            bg="#2d2d2d",
            fg="#000000",
            activebackground="#3d3d3d",
            activeforeground="#000000",
            relief=tk.FLAT,
            padx=10,
            pady=5,
            font=("Consolas", 9)
        )
        multi_pdf_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # テキストファイル追加ボタン（新機能）
        text_file_button = tk.Button(
            button_frame,
            text="テキストファイル追加",
            command=self._select_text_file,
            bg="#2d2d2d",
            fg="#000000",
            activebackground="#3d3d3d",
            activeforeground="#000000",
            relief=tk.FLAT,
            padx=10,
            pady=5,
            font=("Consolas", 9)
        )
        text_file_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # 手動テキスト入力ボタン（新機能）
        manual_text_button = tk.Button(
            button_frame,
            text="テキスト直接入力",
            command=self._open_manual_text_dialog,
            bg="#2d2d2d",
            fg="#000000",
            activebackground="#3d3d3d",
            activeforeground="#000000",
            relief=tk.FLAT,
            padx=10,
            pady=5,
            font=("Consolas", 9)
        )
        manual_text_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # インデックス構築ボタン（新機能）
        # すべてのソースを追加した後、このボタンで検索可能なインデックスを構築
        build_index_button = tk.Button(
            button_frame,
            text="インデックス構築",
            command=self._build_index,
            bg="#238636",
            fg="#000000",
            activebackground="#2ea043",
            activeforeground="#000000",
            relief=tk.FLAT,
            padx=10,
            pady=5,
            font=("Consolas", 9, "bold")
        )
        build_index_button.pack(side=tk.LEFT)
        
        # =========================================
        # ソース一覧表示エリア（新機能・拡張）
        # =========================================
        # 読み込んだソースの一覧を表示し、削除できるエリア
        source_list_frame = tk.Frame(top_frame, bg="#1e1e1e")
        source_list_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # ソース一覧のラベル
        tk.Label(
            source_list_frame,
            text="読み込んだソース:",
            bg="#1e1e1e",
            fg="#000000",
            font=("Consolas", 9)
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        # ソース一覧を表示するリストボックス（削除機能付き）
        # Listbox = リスト形式で項目を表示し、選択できるウィジェット
        listbox_frame = tk.Frame(source_list_frame, bg="#1e1e1e")
        listbox_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.source_listbox = tk.Listbox(
            listbox_frame,
            height=1,  # 高さ1行（スクロール可能）
            bg="#0d1117",
            fg="#c9d1d9",
            selectbackground="#264f78",
            selectforeground="#ffffff",
            font=("Consolas", 8),
            relief=tk.FLAT,
            borderwidth=1,
            highlightthickness=0
        )
        self.source_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # リストボックスのスクロールバー
        source_scrollbar = tk.Scrollbar(
            listbox_frame,
            orient=tk.VERTICAL,
            command=self.source_listbox.yview,
            bg="#1e1e1e",
            troughcolor="#0d1117",
            activebackground="#3d3d3d"
        )
        source_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.source_listbox.config(yscrollcommand=source_scrollbar.set)
        
        # ソース削除ボタン（新機能）
        remove_source_button = tk.Button(
            source_list_frame,
            text="削除",
            command=self._remove_selected_source,
            bg="#da3633",
            fg="#000000",
            activebackground="#f85149",
            activeforeground="#000000",
            relief=tk.FLAT,
            padx=10,
            pady=2,
            font=("Consolas", 8)
        )
        remove_source_button.pack(side=tk.LEFT)
        
        # =========================================
        # 検索重みの設定（Semantic/Keyword）
        # =========================================
        # 検索重み = セマンティック検索とキーワード検索の比重を調整する設定
        # セマンティック検索 = 文章の意味を理解して検索（類似した意味の文書を探す）
        # キーワード検索 = 単語の一致で検索（特定の単語が含まれる文書を探す）
        weight_frame = tk.Frame(top_frame, bg="#1e1e1e")
        weight_frame.pack(side=tk.RIGHT)  # 右側に配置
        
        # Semantic検索の重みラベル
        tk.Label(
            weight_frame,
            text="Semantic:",  # ラベルテキスト
            bg="#1e1e1e",
            fg="#000000",
            font=("Consolas", 9)
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        # Semantic検索の重みを調整するスライダー
        # DoubleVar = 浮動小数点数の値を保持する変数
        self.semantic_weight = tk.DoubleVar(value=0.5)  # 初期値0.5（50%）
        semantic_scale = tk.Scale(
            weight_frame,
            from_=0.0,  # 最小値
            to=1.0,  # 最大値
            resolution=0.1,  # 刻み幅（0.1刻み）
            orient=tk.HORIZONTAL,  # 横方向のスライダー
            variable=self.semantic_weight,  # 値を保持する変数
            bg="#2d2d2d",  # 背景色
            fg="#000000",  # 文字色
            troughcolor="#1e1e1e",  # スライダーのトラックの色
            activebackground="#3d3d3d",  # ドラッグ中の色
            length=100,  # スライダーの長さ
            font=("Consolas", 8)
        )
        semantic_scale.pack(side=tk.LEFT, padx=(0, 10))
        
        # Keyword検索の重みラベル
        tk.Label(
            weight_frame,
            text="Keyword:",  # ラベルテキスト
            bg="#1e1e1e",
            fg="#000000",
            font=("Consolas", 9)
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        # Keyword検索の重みを調整するスライダー
        self.keyword_weight = tk.DoubleVar(value=0.5)  # 初期値0.5（50%）
        keyword_scale = tk.Scale(
            weight_frame,
            from_=0.0,
            to=1.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.keyword_weight,
            bg="#2d2d2d",
            fg="#000000",
            troughcolor="#1e1e1e",
            activebackground="#3d3d3d",
            length=100,
            font=("Consolas", 8)
        )
        keyword_scale.pack(side=tk.LEFT)
        
        # =========================================
        # 中央：チャット表示エリア（ターミナル風）
        # =========================================
        # チャットの会話を表示するエリア
        chat_frame = tk.Frame(main_frame, bg="#1e1e1e")
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))  # 縦方向に拡張
        
        # =========================================
        # スクロール可能なテキストエリア
        # =========================================
        # ScrolledText = スクロールバー付きのテキストエリア
        # 会話履歴が長くなっても、スクロールして見ることができる
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,  # 単語の境界で改行（単語が途中で切れない）
            bg="#0d1117",  # 背景色（GitHub風のダークテーマ）
            fg="#c9d1d9",  # 文字色（ライトグレー）
            insertbackground="#c9d1d9",  # カーソルの色
            selectbackground="#264f78",  # 選択時の背景色（青）
            selectforeground="#ffffff",  # 選択時の文字色（白）
            font=("Consolas", 11),  # フォント（Consolas、サイズ11）
            relief=tk.FLAT,  # 枠線なし
            borderwidth=0,  # 境界線の幅
            padx=15,  # 横方向の内側の余白
            pady=15,  # 縦方向の内側の余白
            state=tk.DISABLED  # 初期状態では編集不可（プログラムからのみ編集可能）
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)  # 縦横方向に拡張
        
        # =========================================
        # ターミナル風のスタイルを設定
        # =========================================
        # tag_configure = テキストに適用するスタイル（色、フォントなど）を定義
        # メッセージの種類（system, user, assistant, error）ごとに異なる色を設定
        
        # システムメッセージ（青、太字）
        self.chat_display.tag_configure("system", foreground="#58a6ff", font=("Consolas", 11, "bold"))
        # ユーザーメッセージ（ライトブルー）
        self.chat_display.tag_configure("user", foreground="#79c0ff", font=("Consolas", 11))
        # アシスタントメッセージ（パステルブルー）
        self.chat_display.tag_configure("assistant", foreground="#a5d6ff", font=("Consolas", 11))
        # エラーメッセージ（赤）
        self.chat_display.tag_configure("error", foreground="#f85149", font=("Consolas", 11))
        # タイムスタンプ（グレー、小さめのフォント）
        self.chat_display.tag_configure("timestamp", foreground="#6e7681", font=("Consolas", 9))
        
        # =========================================
        # 下部：入力エリア
        # =========================================
        # ユーザーが質問を入力するエリア
        input_frame = tk.Frame(main_frame, bg="#1e1e1e")
        input_frame.pack(fill=tk.X)  # 横方向に拡張
        
        # =========================================
        # 入力フィールド（テキスト入力エリア）
        # =========================================
        # Text = 複数行のテキストを入力できるウィジェット
        self.input_field = tk.Text(
            input_frame,
            height=3,  # 高さ（3行分）
            bg="#0d1117",  # 背景色（ダークテーマ）
            fg="#c9d1d9",  # 文字色（ライトグレー）
            insertbackground="#c9d1d9",  # カーソルの色
            selectbackground="#264f78",  # 選択時の背景色
            selectforeground="#ffffff",  # 選択時の文字色
            font=("Consolas", 11),  # フォント
            relief=tk.FLAT,  # 枠線なし
            padx=10,  # 横方向の内側の余白
            pady=10,  # 縦方向の内側の余白
            wrap=tk.WORD  # 単語の境界で改行
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # =========================================
        # キーボードイベントの設定
        # =========================================
        # bind = キーボードやマウスのイベントに関数を紐付ける
        # Enterキーで送信（Shift+Enterで改行）
        self.input_field.bind("<Return>", self._on_enter_key)  # Enterキーが押されたとき
        self.input_field.bind("<Shift-Return>", self._on_shift_enter)  # Shift+Enterが押されたとき
        
        # =========================================
        # 送信ボタン
        # =========================================
        # 質問を送信するボタン
        send_button = tk.Button(
            input_frame,
            text="送信",  # ボタンのテキスト
            command=self._send_message,  # クリック時に実行するメソッド
            bg="#238636",  # 背景色（緑）
            fg="#000000",  # 文字色（黒）
            activebackground="#2ea043",  # マウスオーバー時の背景色
            activeforeground="#000000",  # マウスオーバー時の文字色
            relief=tk.FLAT,  # 枠線なし
            padx=20,  # 横方向の内側の余白
            pady=10,  # 縦方向の内側の余白
            font=("Consolas", 10, "bold"),  # フォント（太字）
            cursor="hand2"  # マウスカーソル（手の形）
        )
        send_button.pack(side=tk.RIGHT)  # 右側に配置
        
        # =========================================
        # クリアボタン
        # =========================================
        # チャット履歴をクリアするボタン
        clear_button = tk.Button(
            input_frame,
            text="クリア",  # ボタンのテキスト
            command=self._clear_chat,  # クリック時に実行するメソッド
            bg="#da3633",  # 背景色（赤）
            fg="#000000",  # 文字色（黒）
            activebackground="#f85149",  # マウスオーバー時の背景色
            activeforeground="#000000",  # マウスオーバー時の文字色
            relief=tk.FLAT,  # 枠線なし
            padx=15,  # 横方向の内側の余白
            pady=10,  # 縦方向の内側の余白
            font=("Consolas", 10),  # フォント
            cursor="hand2"  # マウスカーソル（手の形）
        )
        clear_button.pack(side=tk.RIGHT, padx=(0, 10))  # 右側に配置、左側に10pxの余白
        
        # =========================================
        # 初期フォーカスの設定
        # =========================================
        # アプリケーション起動時に、入力フィールドにフォーカスを設定
        # これにより、すぐにキーボードで入力できる
        self.input_field.focus_set()
    
    def _add_system_message(self, message: str):
        """
        システムメッセージを追加するメソッド
        
        システムからの通知メッセージ（例：「PDFファイルを読み込んでいます」）
        をチャット表示エリアに追加します。
        
        Args:
            message: 表示するメッセージ（文字列）
        """
        # テキストエリアを編集可能にする（初期状態はDISABLED）
        self.chat_display.config(state=tk.NORMAL)
        
        # 現在時刻を取得してタイムスタンプとして表示
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # タイムスタンプを追加（"timestamp"タグでスタイルを適用）
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
        
        # システムメッセージを追加（"system"タグでスタイルを適用）
        self.chat_display.insert(tk.END, f"SYSTEM: {message}\n\n", "system")
        
        # テキストエリアを編集不可に戻す（ユーザーが直接編集できないように）
        self.chat_display.config(state=tk.DISABLED)
        
        # 最新のメッセージまで自動的にスクロール
        self.chat_display.see(tk.END)
    
    def _add_user_message(self, message: str):
        """
        ユーザーメッセージを追加するメソッド
        
        ユーザーが入力した質問をチャット表示エリアに追加します。
        
        Args:
            message: ユーザーのメッセージ（質問文）
        """
        self.chat_display.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
        # "user"タグでスタイルを適用（ライトブルーの色）
        self.chat_display.insert(tk.END, f"USER: {message}\n\n", "user")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def _add_assistant_message(self, message: str):
        """
        アシスタントメッセージを追加するメソッド
        
        LLMが生成した回答をチャット表示エリアに追加します。
        
        Args:
            message: アシスタントのメッセージ（LLMの回答）
        """
        self.chat_display.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
        # "assistant"タグでスタイルを適用（パステルブルーの色）
        self.chat_display.insert(tk.END, f"ASSISTANT: {message}\n\n", "assistant")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def _add_error_message(self, message: str):
        """
        エラーメッセージを追加するメソッド
        
        エラーが発生した場合に、エラーメッセージをチャット表示エリアに追加します。
        
        Args:
            message: エラーメッセージ（エラーの内容）
        """
        self.chat_display.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
        # "error"タグでスタイルを適用（赤色）
        self.chat_display.insert(tk.END, f"ERROR: {message}\n\n", "error")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def _select_pdf_file(self):
        """
        PDFファイルを選択するメソッド（後方互換性のため維持）
        
        ファイル選択ダイアログを開いて、ユーザーにPDFファイルを選択させます。
        選択されたファイルは、_add_pdf_to_system()メソッドで読み込まれます。
        """
        # filedialog.askopenfilename = ファイル選択ダイアログを表示
        file_path = filedialog.askopenfilename(
            title="PDFファイルを選択",  # ダイアログのタイトル
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]  # 表示するファイルタイプ
        )
        
        # ファイルが選択された場合（キャンセルされた場合は空文字列が返る）
        if file_path:
            self._add_pdf_to_system(file_path)  # PDFファイルを追加
    
    def _select_multiple_pdfs(self):
        """
        複数のPDFファイルを選択するメソッド（新機能）
        
        ファイル選択ダイアログを開いて、ユーザーに複数のPDFファイルを選択させます。
        選択されたすべてのファイルが、MultiSourceRagSystemに追加されます。
        """
        # filedialog.askopenfilenames = 複数ファイル選択ダイアログを表示
        file_paths = filedialog.askopenfilenames(
            title="複数のPDFファイルを選択",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        # 選択されたファイルがある場合
        if file_paths:
            # 各PDFファイルを追加
            for file_path in file_paths:
                self._add_pdf_to_system(file_path)
    
    def _select_text_file(self):
        """
        テキストファイルを選択するメソッド（新機能）
        
        ファイル選択ダイアログを開いて、ユーザーにテキストファイル（.txt）を選択させます。
        選択されたファイルが、MultiSourceRagSystemに追加されます。
        """
        # テキストファイル選択ダイアログを表示
        file_path = filedialog.askopenfilename(
            title="テキストファイルを選択",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        # ファイルが選択された場合
        if file_path:
            self._add_text_file_to_system(file_path)
    
    def _open_manual_text_dialog(self):
        """
        手動テキスト入力ダイアログを開くメソッド（新機能）
        
        新しいウィンドウを開いて、ユーザーにテキストを直接入力させます。
        入力されたテキストが、MultiSourceRagSystemに追加されます。
        """
        # 新しいウィンドウ（ダイアログ）を作成
        dialog = tk.Toplevel(self.root)
        dialog.title("テキスト直接入力")
        dialog.geometry("600x500")
        dialog.configure(bg="#1e1e1e")
        
        # ソース名入力エリア
        name_frame = tk.Frame(dialog, bg="#1e1e1e")
        name_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            name_frame,
            text="ソース名:",
            bg="#1e1e1e",
            fg="#000000",
            font=("Consolas", 10)
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        name_entry = tk.Entry(
            name_frame,
            bg="#0d1117",
            fg="#c9d1d9",
            insertbackground="#c9d1d9",
            font=("Consolas", 10),
            width=30
        )
        name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        name_entry.insert(0, f"手動入力_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # テキスト入力エリア
        text_frame = tk.Frame(dialog, bg="#1e1e1e")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        tk.Label(
            text_frame,
            text="テキスト内容:",
            bg="#1e1e1e",
            fg="#000000",
            font=("Consolas", 10)
        ).pack(anchor=tk.W, pady=(0, 5))
        
        text_area = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            bg="#0d1117",
            fg="#c9d1d9",
            insertbackground="#c9d1d9",
            font=("Consolas", 10),
            height=15
        )
        text_area.pack(fill=tk.BOTH, expand=True)
        
        # ボタンフレーム
        button_frame = tk.Frame(dialog, bg="#1e1e1e")
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        def add_text():
            """テキストを追加する内部関数"""
            source_name = name_entry.get().strip()
            text_content = text_area.get("1.0", tk.END).strip()
            
            if not source_name:
                messagebox.showerror("エラー", "ソース名を入力してください。")
                return
            
            if not text_content:
                messagebox.showerror("エラー", "テキスト内容を入力してください。")
                return
            
            # テキストを追加
            self._add_manual_text_to_system(text_content, source_name)
            # ダイアログを閉じる
            dialog.destroy()
        
        # 追加ボタン
        add_button = tk.Button(
            button_frame,
            text="追加",
            command=add_text,
            bg="#238636",
            fg="#000000",
            activebackground="#2ea043",
            activeforeground="#000000",
            relief=tk.FLAT,
            padx=20,
            pady=5,
            font=("Consolas", 10, "bold")
        )
        add_button.pack(side=tk.RIGHT, padx=(10, 0))
        
        # キャンセルボタン
        cancel_button = tk.Button(
            button_frame,
            text="キャンセル",
            command=dialog.destroy,
            bg="#da3633",
            fg="#000000",
            activebackground="#f85149",
            activeforeground="#000000",
            relief=tk.FLAT,
            padx=20,
            pady=5,
            font=("Consolas", 10)
        )
        cancel_button.pack(side=tk.RIGHT)
        
        # フォーカスをテキストエリアに設定
        text_area.focus_set()
    
    def _add_pdf_to_system(self, pdf_path: str):
        """
        PDFファイルをMultiSourceRagSystemに追加するメソッド（新機能）
        
        Args:
            pdf_path: PDFファイルのパス
        """
        # MultiSourceRagSystemが初期化されていない場合は初期化
        if self.multi_source_system is None:
            self.multi_source_system = MultiSourceRagSystem(persist_dir="./chroma_db")
        
        try:
            # PDFファイルを追加
            self.multi_source_system.add_pdf(pdf_path)
            
            # ソース一覧を更新
            self._update_source_list()
            
            # 成功メッセージを表示
            self._add_system_message(f"PDFファイルを追加しました: {os.path.basename(pdf_path)}")
        except Exception as e:
            self._add_error_message(f"PDFファイルの追加に失敗しました: {str(e)}")
    
    def _add_text_file_to_system(self, text_path: str):
        """
        テキストファイルをMultiSourceRagSystemに追加するメソッド（新機能）
        
        Args:
            text_path: テキストファイルのパス
        """
        # MultiSourceRagSystemが初期化されていない場合は初期化
        if self.multi_source_system is None:
            self.multi_source_system = MultiSourceRagSystem(persist_dir="./chroma_db")
        
        try:
            # テキストファイルを追加
            self.multi_source_system.add_text_file(text_path)
            
            # ソース一覧を更新
            self._update_source_list()
            
            # 成功メッセージを表示
            self._add_system_message(f"テキストファイルを追加しました: {os.path.basename(text_path)}")
        except Exception as e:
            self._add_error_message(f"テキストファイルの追加に失敗しました: {str(e)}")
    
    def _add_manual_text_to_system(self, text_content: str, source_name: str):
        """
        手動テキストをMultiSourceRagSystemに追加するメソッド（新機能）
        
        Args:
            text_content: テキストの内容
            source_name: ソース名（識別名）
        """
        # MultiSourceRagSystemが初期化されていない場合は初期化
        if self.multi_source_system is None:
            self.multi_source_system = MultiSourceRagSystem(persist_dir="./chroma_db")
        
        try:
            # 手動テキストを追加
            self.multi_source_system.add_manual_text(text_content, source_name)
            
            # ソース一覧を更新
            self._update_source_list()
            
            # 成功メッセージを表示
            self._add_system_message(f"テキストを追加しました: {source_name}")
        except Exception as e:
            self._add_error_message(f"テキストの追加に失敗しました: {str(e)}")
    
    def _update_source_list(self):
        """
        ソース一覧を更新するメソッド（新機能・拡張）
        
        MultiSourceRagSystemに追加されたソースの一覧をUIのリストボックスに表示します。
        削除機能に対応するため、リストボックス形式に変更しました。
        """
        if self.multi_source_system is None:
            self.source_listbox.delete(0, tk.END)
            self.source_listbox.insert(0, "なし")
            return
        
        sources = self.multi_source_system.get_source_info()
        
        # リストボックスをクリア
        self.source_listbox.delete(0, tk.END)
        
        if not sources:
            self.source_listbox.insert(0, "なし")
            return
        
        # ソース一覧をリストボックスに追加
        for i, source in enumerate(sources):
            source_type_label = {
                "pdf": "PDF",
                "text_file": "TXT",
                "manual_text": "手動"
            }.get(source["type"], "不明")
            
            # リストボックスに表示する形式: "[PDF] ファイル名"
            display_text = f"[{source_type_label}] {source['name']}"
            self.source_listbox.insert(tk.END, display_text)
        
        # リストボックスの高さを調整（最大3行まで表示）
        item_count = len(sources)
        self.source_listbox.config(height=min(item_count, 3))
    
    def _remove_selected_source(self):
        """
        選択されたソースを削除するメソッド（新機能）
        
        リストボックスで選択されたソースを削除します。
        削除後は、インデックスを再構築する必要があります。
        """
        if self.multi_source_system is None:
            self._add_error_message("ソースが追加されていません。")
            return
        
        # リストボックスで選択された項目のインデックスを取得
        selected_indices = self.source_listbox.curselection()
        
        if not selected_indices:
            self._add_error_message("削除するソースを選択してください。")
            return
        
        # 選択されたインデックス（リストボックス内の位置）
        listbox_index = selected_indices[0]
        
        # ソース情報を取得
        sources = self.multi_source_system.get_source_info()
        
        if listbox_index >= len(sources):
            self._add_error_message("選択されたソースが見つかりません。")
            return
        
        # 削除するソースの情報を取得
        source_to_remove = sources[listbox_index]
        source_name = source_to_remove["name"]
        source_type = source_to_remove["type"]
        
        try:
            # 確認ダイアログを表示
            source_type_label = {
                "pdf": "PDF",
                "text_file": "テキストファイル",
                "manual_text": "手動テキスト"
            }.get(source_type, "ソース")
            
            confirm = messagebox.askyesno(
                "確認",
                f"{source_type_label}「{source_name}」を削除しますか？\n"
                f"削除後は、インデックスを再構築する必要があります。"
            )
            
            if not confirm:
                return
            
            # ソースを削除
            self.multi_source_system.remove_source(listbox_index)
            
            # ソース一覧を更新
            self._update_source_list()
            
            # RAGシステムをクリア（インデックスを再構築する必要があるため）
            self.rag_system = None
            
            # 成功メッセージを表示
            self._add_system_message(
                f"ソース「{source_name}」を削除しました。\n"
                f"インデックスを再構築するには、「インデックス構築」ボタンを押してください。"
            )
            
        except Exception as e:
            self._add_error_message(f"ソースの削除に失敗しました: {str(e)}")
    
    def _build_index(self):
        """
        インデックスを構築するメソッド（新機能）
        
        すべてのソースを追加した後、このメソッドを呼び出すことで、
        検索可能なインデックス（BM25とベクトルDB）を構築します。
        """
        if self.multi_source_system is None:
            self._add_error_message("ソースが追加されていません。先にソースを追加してください。")
            return
        
        if not self.multi_source_system.docs:
            self._add_error_message("チャンクが存在しません。先にソースを追加してください。")
            return
        
        try:
            self._add_system_message("インデックスを構築しています...")
            self.root.update()
            
            # インデックス構築処理を別スレッドで実行
            def build_thread():
                try:
                    # インデックスを構築
                    self.multi_source_system.build_index()
                    
                    # ReRankingRAGシステムを初期化
                    self.rag_system = MultiSourceReRankingRAG(self.multi_source_system)
                    
                    # プロンプトテンプレートを設定
                    self.rag_system.prompt_template = self.prompt_template
                    
                    # UIスレッドでメッセージを更新
                    sources = self.multi_source_system.get_source_info()
                    total_chunks = len(self.multi_source_system.docs)
                    
                    self.root.after(0, lambda: self._add_system_message(
                        f"インデックス構築が完了しました！\n"
                        f"ソース数: {len(sources)}\n"
                        f"総チャンク数: {total_chunks}\n"
                        f"質問を開始できます。"
                    ))
                except Exception as e:
                    self.root.after(0, lambda: self._add_error_message(
                        f"インデックス構築に失敗しました: {str(e)}"
                    ))
            
            thread = threading.Thread(target=build_thread, daemon=True)
            thread.start()
            
        except Exception as e:
            self._add_error_message(f"エラーが発生しました: {str(e)}")
    
    def _load_pdf(self, file_path: str):
        """
        PDFファイルを読み込むメソッド（後方互換性のため維持）
        
        このメソッドは、既存のコードとの互換性のために残されています。
        新しいコードでは、_add_pdf_to_system()と_build_index()を使用してください。
        
        Args:
            file_path: PDFファイルのパス（文字列）
        """
        # 新しい方式に移行（後方互換性のため）
        self._add_pdf_to_system(file_path)
        
        # 自動的にインデックスを構築（既存の動作を維持）
        if self.multi_source_system and self.multi_source_system.docs:
            self._build_index()
    
    def _on_enter_key(self, event):
        """
        Enterキーが押されたときの処理
        
        このメソッドは、ユーザーがEnterキーを押したときに呼ばれます。
        - Enterキーのみ: メッセージを送信
        - Shift+Enter: 改行（通常の動作）
        
        Args:
            event: イベントオブジェクト（押されたキーの情報などが含まれる）
        """
        # Shift+Enterの場合は改行を許可（通常の動作）
        # event.state & 0x1 = Shiftキーが押されているかどうかをチェック
        if event.state & 0x1:  # Shiftキーが押されている
            return None  # Noneを返す = 通常の動作（改行）を許可
        
        # Enterキーのみの場合は送信
        self._send_message()
        # "break"を返す = 通常の動作（改行）をキャンセル
        return "break"
    
    def _on_shift_enter(self, event):
        """
        Shift+Enterキーが押されたときの処理（改行）
        
        このメソッドは、ユーザーがShift+Enterキーを押したときに呼ばれます。
        改行を許可するため、Noneを返します（通常の動作を許可）。
        
        Args:
            event: イベントオブジェクト
        """
        return None  # Noneを返す = 通常の動作（改行）を許可
    
    def _send_message(self):
        """
        メッセージを送信するメソッド
        
        このメソッドは、ユーザーが質問を入力して送信したときに呼ばれます。
        
        処理の流れ：
        1. 入力フィールドから質問を取得
        2. ユーザーの質問をチャット表示エリアに表示
        3. RAGシステムで検索を実行（PDFから関連情報を取得）
        4. 検索結果を文脈として、LLMに質問を投げかける
        5. LLMが生成した回答をチャット表示エリアに表示
        """
        # =========================================
        # ステップ1: 入力フィールドからテキストを取得
        # =========================================
        # get("1.0", tk.END) = テキストエリアの最初から最後までを取得
        # "1.0" = 1行目、0文字目（最初の文字）
        # strip() = 前後の空白を削除
        user_input = self.input_field.get("1.0", tk.END).strip()
        
        # 入力が空の場合は何もしない
        if not user_input:
            return
        
        # =========================================
        # ステップ2: 入力フィールドをクリア
        # =========================================
        # 次の質問を入力しやすくするため、入力フィールドを空にする
        self.input_field.delete("1.0", tk.END)
        
        # =========================================
        # ステップ3: ユーザーメッセージを表示
        # =========================================
        # ユーザーが入力した質問をチャット表示エリアに表示
        self._add_user_message(user_input)
        
        # =========================================
        # ステップ4: RAGシステムのチェック
        # =========================================
        # RAGシステムが初期化されていない場合（ソースが読み込まれていない場合）
        if self.rag_system is None:
            self._add_error_message("ソースが読み込まれていません。先にソースを追加して「インデックス構築」ボタンを押してください。")
            return
        
        # =========================================
        # ステップ5: 検索と回答生成を別スレッドで実行
        # =========================================
        # なぜ別スレッド？
        # - 検索とLLMの回答生成は時間がかかる処理
        # - メインスレッド（UIスレッド）で実行すると、UIがフリーズする
        # - 別スレッドで実行することで、UIを操作し続けられる
        def process_question():
            try:
                # =========================================
                # ステップ5-1: 検索重みを取得
                # =========================================
                # 検索重み = セマンティック検索とキーワード検索の比重
                # ユーザーがUIで設定した値を取得
                w_sem = self.semantic_weight.get()  # セマンティック検索の重み（0.0〜1.0）
                w_key = self.keyword_weight.get()  # キーワード検索の重み（0.0〜1.0）
                
                # =========================================
                # ステップ5-2: 検索を実行
                # =========================================
                # 検索中メッセージを表示
                self.root.after(0, lambda: self._add_system_message("検索中..."))
                
                # RAGシステムで検索を実行
                # search() = PDFから質問に関連する文書を検索する
                top_docs = self.rag_system.search(
                    question=user_input,  # ユーザーの質問
                    k=5,  # 返す文書の数（上位5件）
                    w_sem=w_sem,  # セマンティック検索の重み
                    w_key=w_key,  # キーワード検索の重み
                    candidate_k=60  # リランキング前の候補数
                )
                
                # =========================================
                # ステップ5-3: 検索結果を文脈として結合（ソース情報付き）
                # =========================================
                # 検索結果の各文書の内容を改行2つで区切って結合
                # 各チャンクにソース情報（source_type, source_name）が含まれている
                # これにより、どのソースから来た情報かが分かる
                context_parts = []
                for doc in top_docs:
                    # ソース情報を取得（metadataから）
                    source_type = doc.metadata.get('source_type', 'unknown')
                    source_name = doc.metadata.get('source_name', 'unknown')
                    
                    # ソース情報を付けて文脈に追加
                    # 例：「[PDF: linuxtext_ver4.0.0.pdf]\n文書の内容...」
                    source_label = {
                        'pdf': 'PDF',
                        'text_file': 'TXT',
                        'manual_text': '手動'
                    }.get(source_type, source_type.upper())
                    
                    context_parts.append(f"[{source_label}: {source_name}]\n{doc.page_content}")
                
                # すべての検索結果を結合（これがLLMに渡す「文脈」になる）
                context = "\n\n".join(context_parts)
                
                # デバッグ用：検索結果のソース情報を表示（オプション）
                # どのソースから情報が得られたかを確認できる
                source_summary = {}
                for doc in top_docs:
                    source_name = doc.metadata.get('source_name', 'unknown')
                    source_summary[source_name] = source_summary.get(source_name, 0) + 1
                
                if source_summary:
                    sources_used = ", ".join([f"{name}({count}件)" for name, count in source_summary.items()])
                    self.root.after(0, lambda: self._add_system_message(f"検索結果: {sources_used}から情報を取得"))
                
                # =========================================
                # ステップ5-4: プロンプトを生成
                # =========================================
                # プロンプトテンプレートに、検索結果（context）と質問（question）を埋め込む
                # これがLLMに渡す最終的な指示文になる
                prompt = self.prompt_template.format(
                    context=context,  # 検索結果（PDFの内容）
                    question=user_input  # ユーザーの質問
                )
                
                # =========================================
                # ステップ5-5: LLMにプロンプトを渡して回答を生成
                # =========================================
                # 回答生成中メッセージを表示
                self.root.after(0, lambda: self._add_system_message("回答を生成中..."))
                
                # LLMにプロンプトを渡して回答を生成
                # invoke() = LLMを呼び出して回答を取得
                # 使用しているLLM: Ollamaのllama3（pdf_rag.pyで設定）
                raw_answer = self.rag_system.llm.invoke(prompt)
                
                # =========================================
                # ステップ5-6: 回答を表示
                # =========================================
                # LLMが生成した回答をチャット表示エリアに表示
                self.root.after(0, lambda: self._add_assistant_message(raw_answer))
                
                # =========================================
                # ステップ5-7: 会話履歴に追加
                # =========================================
                # 会話履歴に、ユーザーの質問とアシスタントの回答を記録
                # 将来、会話履歴を活用する機能を追加する場合に備えて保存
                self.conversation_history.append({
                    "role": "user",  # ユーザーの質問
                    "message": user_input,
                    "timestamp": datetime.now().isoformat()  # タイムスタンプ
                })
                self.conversation_history.append({
                    "role": "assistant",  # アシスタントの回答
                    "message": raw_answer,
                    "timestamp": datetime.now().isoformat()  # タイムスタンプ
                })
                
            except Exception as e:
                # エラーが発生した場合、エラーメッセージを表示
                self.root.after(0, lambda: self._add_error_message(
                    f"エラーが発生しました: {str(e)}"
                ))
        
        # 別スレッドを開始
        # daemon=True = メインプログラムが終了したら、このスレッドも終了する
        thread = threading.Thread(target=process_question, daemon=True)
        thread.start()
    
    def _clear_chat(self):
        """
        チャット履歴をクリアするメソッド
        
        チャット表示エリアの内容と、会話履歴をクリアします。
        これにより、新しい会話を始められます。
        """
        # テキストエリアを編集可能にする
        self.chat_display.config(state=tk.NORMAL)
        # テキストエリアの内容をすべて削除
        self.chat_display.delete("1.0", tk.END)
        # テキストエリアを編集不可に戻す
        self.chat_display.config(state=tk.DISABLED)
        # 会話履歴のリストをクリア
        self.conversation_history.clear()
        # クリア完了メッセージを表示
        self._add_system_message("チャット履歴をクリアしました。")


def main():
    """
    メイン関数
    
    この関数は、アプリケーションを起動する際に最初に呼ばれます。
    Tkinterのウィンドウを作成し、RAGTerminalUIクラスのインスタンスを作成して、
    アプリケーションを実行します。
    """
    # =========================================
    # Tkinterのルートウィンドウを作成
    # =========================================
    # Tk() = Tkinterのメインウィンドウを作成
    root = tk.Tk()
    
    # =========================================
    # RAGTerminalUIクラスのインスタンスを作成
    # =========================================
    # RAGTerminalUI = このアプリケーションのメインクラス
    # このクラスがUIの構築と、RAGシステムとの連携を行います
    app = RAGTerminalUI(root)
    
    # =========================================
    # メインループを開始
    # =========================================
    # mainloop() = アプリケーションのメインループを開始
    # このループが、ユーザーの操作（ボタンクリック、キー入力など）を待ち受けます
    # ウィンドウが閉じられるまで、このループが実行され続けます
    root.mainloop()


if __name__ == "__main__":
    """
    このスクリプトが直接実行された場合のみ、main()関数を呼び出す
    
    if __name__ == "__main__": の意味：
    - このファイルが直接実行された場合: __name__ は "__main__" になる
    - このファイルが他のファイルからインポートされた場合: __name__ は "rag_terminal_ui" になる
    
    これにより、このファイルをインポートしても、main()が自動的に実行されない
    """
    main()

