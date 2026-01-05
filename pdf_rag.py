"""
PDF RAGシステム（Retrieval-Augmented Generation）

このプログラムは、PDFファイルから情報を検索して質問に答えるシステムです。
主な機能：
1. PDFを読み込んで検索可能な形式に変換（チャンク分割、ベクトル化）
2. キーワード検索（BM25）と意味検索（セマンティック検索）を組み合わせたハイブリッド検索
3. 検索結果をリランキングして精度を向上
4. LLMを使って質問に対する回答を生成

用語説明：
- RAG: Retrieval-Augmented Generation（検索拡張生成）
- チャンク: 文書を小さな断片に分割したもの
- ベクトル化: 文章を数値の配列（ベクトル）に変換すること
- BM25: キーワードマッチングに基づく検索アルゴリズム
- セマンティック検索: 文章の意味を理解して検索する方法
- リランキング: 検索結果の順位を再評価して並び替えること
"""

# =========================================
# ライブラリのインポート
# =========================================
# PDFファイルを読み込むためのライブラリ
from langchain_community.document_loaders import PyPDFLoader

# テキストを適切なサイズに分割するためのライブラリ
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ベクトルデータベース（検索用のデータベース）
from langchain_community.vectorstores import Chroma

# 文章をベクトル（数値の配列）に変換するためのライブラリ
from langchain_community.embeddings import HuggingFaceEmbeddings

# LLM（大規模言語モデル）を呼び出すためのライブラリ
from langchain_community.llms import Ollama

# 複数の検索方法を組み合わせるためのライブラリ
from langchain.retrievers import EnsembleRetriever

# キーワード検索（BM25）のためのライブラリ
from langchain_community.retrievers import BM25Retriever

# 文書データを表現するためのクラス
from langchain.docstore.document import Document

# プロンプトテンプレート（LLMへの指示文を管理）
from langchain.prompts import PromptTemplate

# リランキング用のモデル（検索結果の精度を向上させる）
from sentence_transformers import CrossEncoder

# 型ヒント（変数の型を明示するための機能）
from typing import List, Tuple

# 正規表現、OS操作、警告制御のための標準ライブラリ
import re, os, warnings

# 警告ログを抑制（検索スコアなど不要なログを出さないため）
# これにより、実行時の出力がすっきりします
warnings.filterwarnings("ignore")

# =========================================
# ① PDF読み込み＆チャンク化の箱（インポートフェーズ）
# =========================================
class PDFRagSystem:
    """
    PDFファイルを読み込んで、検索可能な形式に変換するクラス
    
    役割：
    - PDFをテキストに変換
    - テキストを適切なサイズのチャンク（断片）に分割
    - 384次元のベクトル化（意味検索用）
    - キーワード検索用のBM25辞書を作成
    - Chroma DBに保存（次回以降の検索で再利用可能）
    """
    def __init__(self, persist_dir="./chroma_db"):
        """
        初期化メソッド
        
        Args:
            persist_dir: ベクトルデータベースを保存するフォルダのパス
                        デフォルトは "./chroma_db"（現在のフォルダ内に作成）
        """
        # Chroma DBの保存フォルダを1つの変数で管理（統一して使い回す）
        # このフォルダにベクトルデータベースが保存され、次回実行時に再利用できます
        self.persist_dir = persist_dir
        
        # ベクトルストア（後で初期化される）
        # 文書をベクトル化して保存するデータベース
        self.vectorstore = None
        
        # チャンク化された文書のリスト
        # Document型のオブジェクトのリストとして保持
        self.docs: List[Document] = []
        
        # Semantic検索用のembeddingモデル（384次元で統一）
        # embedding = 文章を数値の配列（ベクトル）に変換すること
        # このモデルは文章の意味を384個の数値で表現します
        # 例：「猫」と「ネコ」は似た意味なので、似たベクトルになります
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Keyword検索用のBM25は仕込み工程で1回だけ作る
        # BM25 = キーワードマッチングに基づく検索アルゴリズム
        # （検索時には再構築せず、このインスタンスを再利用する）
        # 初期状態ではNoneで、PDFを読み込んだ後に作成されます
        self.bm25 = None

    def clean_text(self, text: str) -> str:
        """
        テキストのノイズを除去する関数
        
        PDFから読み込んだテキストには、不要な改行や空白が含まれることがあります。
        この関数は、それらを整理して読みやすい形式にします。
        
        Args:
            text: クリーニング前のテキスト
            
        Returns:
            クリーニング後のテキスト
            
        例：
            入力: "あいうえお\n\n\nかきくけこ"
            出力: "あいうえお\n\nかきくけこ"（連続する3つ以上の改行を2つに統一）
        """
        # 正規表現を使って、3つ以上の連続する改行を2つの改行に置き換える
        # r"\n{3,}" は「改行が3つ以上連続している部分」を意味する
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # 前後の空白を削除して返す
        return text.strip()

    def import_pdf(self, pdf_path: str):
        """
        PDFファイルを読み込んで、検索可能な形式に変換する
        
        処理の流れ：
        1. PDFをテキストに変換
        2. テキストを適切なサイズのチャンクに分割
        3. 短すぎるチャンクを除外
        4. BM25検索器を構築（キーワード検索用）
        5. Chroma DBを生成（意味検索用のベクトルDB）
        """
        # =========================================
        # ステップ1: PDFをテキストに変換
        # =========================================
        # PDFファイルを読み込んで、ページごとに分割されたDocumentオブジェクトのリストを取得
        # 例：100ページのPDFなら、100個のDocumentが返ってくる
        raw_docs = PyPDFLoader(pdf_path).load()

        # =========================================
        # ステップ2: テキストを適切なサイズのチャンクに分割
        # =========================================
        # なぜチャンクに分割するのか？
        # - 長い文書をそのまま検索すると、関連部分が埋もれてしまう
        # - 小さなチャンクに分けることで、より正確に検索できる
        # - ただし、小さすぎると文脈が失われるので、適切なサイズが重要
        
        # チャンキング戦略の設定
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=420,      # 1つのチャンクの最大文字数（420文字）
            chunk_overlap=80,     # チャンク間の重複文字数（80文字）
                                 # 重複があることで、文脈が途切れにくくなる
            separators=["\n\n", "\n", "。", ".", " ", ""]
                                 # 分割する際の優先順位
                                 # 1. 段落（\n\n）で分割を試みる
                                 # 2. ダメなら改行（\n）で分割
                                 # 3. それでもダメなら句点（。や.）で分割
                                 # 4. 最後の手段として空白や文字単位で分割
        )

        # PDFをチャンクに分割
        # 例：100ページのPDFが、500個のチャンクに分割される
        chunks = splitter.split_documents(raw_docs)

        # =========================================
        # ステップ3: チャンクをクリーニングして整理
        # =========================================
        # クリーニング後のチャンクをDocument形式で保存
        # 各チャンクに一意のIDを付与（リランキング前後の順番比較に使用）
        cleaned = []
        for idx, c in enumerate(chunks):
            # テキストをクリーニング（不要な改行などを削除）
            text = self.clean_text(c.page_content)
            
            # 文章が存在し、短すぎないチャンクだけ採用（120文字以上）
            # 短すぎるチャンクは情報が少なすぎて検索に役立たないため除外
            if text and len(text) > 120:
                # metadataにチャンク識別子を追加
                # 例：3ページ目の5番目のチャンク → "chunk_3_5"
                chunk_id = f"chunk_{c.metadata.get('page', 0)}_{idx}"
                metadata = c.metadata.copy()
                metadata['chunk_id'] = chunk_id
                cleaned.append(Document(page_content=text, metadata=metadata))

        # クリーニング済みのチャンクを保存
        self.docs = cleaned

        # =========================================
        # ステップ4: キーワード検索用のBM25検索器を構築
        # =========================================
        # BM25検索器を1回だけ構築（キーワード検索のための辞書作り）
        # BM25とは？
        # - 単語の出現頻度を使って検索する手法
        # - 質問に含まれる単語が多く出現する文書を高く評価
        # - 例：「systemctl」という単語が多く含まれるチャンクを優先的に返す
        self.bm25 = BM25Retriever.from_documents(self.docs)
        self.bm25.k = 60  # 候補取得数（検索時に最大60件の候補を取得）
                          # この値が大きいほど、多様な検索結果が得られる

        # =========================================
        # ステップ5: セマンティック検索用のベクトルDBを生成
        # =========================================
        # Chroma DB の初期化（古いDBが残っていたら削除して新規作成）
        # これにより、次元不一致エラーを防ぐ
        # 注意：既存のDBを削除するので、PDFを再読み込みする場合は注意
        if os.path.exists(self.persist_dir):
            os.system(f"rm -rf {self.persist_dir}")

        # ベクトルDB（Semantic検索用の箱）を生成
        # 各チャンクを384次元のベクトルに変換して保存
        # ベクトル化することで、文章の意味を数値で表現できる
        # 例：「猫」と「ネコ」は似た意味なので、似たベクトルになる
        self.vectorstore = Chroma.from_documents(
            self.docs,                    # チャンク化された文書
            self.embeddings,              # ベクトル化に使うモデル
            persist_directory=self.persist_dir  # 保存先フォルダ
        )
        # これで、次回実行時に既存のDBを読み込んで再利用できます

# =========================================
# ② RAGシステムの箱（検索と回答生成）
# =========================================
class ReRankingRAG:
    """
    リランキング付きRAGシステムのクラス
    
    役割：
    - Semantic検索（意味検索）とBM25検索（キーワード検索）を組み合わせる
    - CrossEncoderを使って検索結果を再ランキング（精度向上）
    - LLMを使って質問に対する結論を生成
    - weightsの比重を変えることで、検索の焦点を調整可能
    """
    def __init__(self, docs: List[Document], persist_dir="./chroma_db"):
        # 検索対象チャンクを保持
        self.docs = docs
        self.persist_dir = persist_dir

        # embeddingモデル（Semantic検索用、384次元で統一）
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # =========================================
        # LLM（大規模言語モデル）の初期化
        # =========================================
        # LLMモデル（Ollama llama3）
        # LLM = 質問に対して文章を生成するAIモデル
        # temperature=0.0で一貫性のある回答を生成
        # （temperatureが高いとランダム性が増し、低いと一貫性が増す）
        self.llm = Ollama(model="llama3:latest", temperature=0.0)

        # =========================================
        # セマンティック検索器の初期化
        # =========================================
        # semantic検索器はすでに作成済みのDBを読むだけ（ここでは新規生成しない）
        # 既存のChroma DBから読み込むことで、次元の一貫性を保つ
        # 次元の一貫性 = ベクトルのサイズ（384次元）が同じであること
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,  # DBの保存先
            embedding_function=self.embeddings    # ベクトル化に使うモデル
        )
        # 検索器（retriever）を作成
        # search_kwargs={"k": 60} = 検索時に最大60件の候補を取得
        self.semantic = self.vectorstore.as_retriever(search_kwargs={"k": 60})

        # =========================================
        # キーワード検索器（BM25）の初期化
        # =========================================
        # Keyword検索器（BM25）はPDF仕込みで作ったものを使うので再構築しない
        # ただし、ReRankingRAGクラス内でも参照できるように再構築が必要
        # 注意：PDFRagSystemで作成したBM25とは別のインスタンス
        self.bm25 = BM25Retriever.from_documents(self.docs)
        self.bm25.k = 60  # 候補取得数（検索時に最大60件の候補を取得）

        # =========================================
        # リランキング用のCrossEncoderの初期化
        # =========================================
        # CrossEncoderリランカー
        # リランキング = 検索結果の順位を再評価して並び替えること
        # CrossEncoderとは？
        # - 質問と文書のペアを同時に入力として受け取り、関連度スコアを計算
        # - より正確な関連度を判定できる（ただし計算コストが高い）
        # - 例：「systemctl」という質問に対して、各チャンクの関連度を0.0〜1.0で評価
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # =========================================
        # プロンプトテンプレートの初期化
        # =========================================
        # プロンプトテンプレート = LLMへの指示文のテンプレート
        # contextとquestionを変数として受け取る
        # {context} と {question} の部分が後で実際の値に置き換えられる
        template = """【参考情報】
{context}

【質問】
{question}

【指示】
- 必ず日本語で回答する
- 1つの軸に偏らず複数の評価軸で比較する
- 余計な説明はせず「結論：〜」の1行だけ出力してください"""
        
        self.prompt_template = PromptTemplate(
            template=template,                    # テンプレート文字列
            input_variables=["context", "question"]  # 変数名のリスト
        )

    def clean_text(self, text: str) -> str:
        """テキストのノイズを除去する関数"""
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def search(self, question: str, k: int, w_sem: float, w_key: float, candidate_k: int = 60) -> List[Document]:
        """
        ハイブリッド検索 + リランキングを実行する
        
        処理の流れ：
        1. Semantic検索とBM25検索をEnsembleRetrieverで統合
        2. weightsの比重（w_sem, w_key）で検索結果を調整
        3. CrossEncoderで各候補を再評価してスコア計算
        4. スコアの高い順に並び替え（リランキング）
        5. 上位k件を返す
        
        Args:
            question: 検索クエリ（質問文）
            k: 返す文書の数
            w_sem: Semantic検索の重み（0.0〜1.0）
            w_key: BM25検索の重み（0.0〜1.0）
            candidate_k: リランキング前の候補数
        
        Returns:
            リランキング後の上位k件の文書リスト
        """
        # =========================================
        # ステップ1: ハイブリッド検索（Semantic + Keyword）
        # =========================================
        # Semantic + Keyword のハイブリッド検索
        # ハイブリッド = 2つの検索方法を組み合わせること
        # EnsembleRetrieverは2つの検索結果を統合する
        # - self.semantic: セマンティック検索（意味検索）
        # - self.bm25: キーワード検索（BM25）
        ensemble = EnsembleRetriever(
            retrievers=[self.semantic, self.bm25],  # 2つの検索器
            weights=[w_sem, w_key]                   # それぞれの重み
        )
        # 重みの例：
        # - w_sem=0.9, w_key=0.1 → セマンティック検索を重視
        # - w_sem=0.1, w_key=0.9 → キーワード検索を重視
        # - w_sem=0.5, w_key=0.5 → 両方を均等に重視
        
        # 検索を実行して候補を取得（最大candidate_k件）
        candidates = ensemble.get_relevant_documents(question)

        # =========================================
        # ステップ2: リランキング（検索結果の再評価）
        # =========================================
        # CrossEncoderでスコア計算し並び替え
        # 質問と各候補文書のペアを作成
        # 例：[("systemctlとは？", "systemctlはサービスを管理するコマンド..."), ...]
        pairs: List[Tuple[str, str]] = [(question, d.page_content) for d in candidates]
        
        # 各ペアの関連度スコアを計算
        # スコアは0.0〜1.0の範囲で、高いほど関連度が高い
        scores = self.reranker.predict(pairs)
        
        # スコアの高い順に並び替え
        # zip(candidates, scores) = [(候補1, スコア1), (候補2, スコア2), ...]
        # key=lambda x: x[1] = スコア（2番目の要素）でソート
        # reverse=True = 降順（高い順）
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        # 上位k件を返す（スコアは返さない）
        # ranked[:k] = 上位k件を取得
        # [doc for doc, _score in ...] = スコアを捨てて、文書だけを返す
        return [doc for doc, _score in ranked[:k]]

    def answer(self, question: str, k: int, w_sem: float, w_key: float, candidate_k: int = 60) -> str:
        """
        質問に対する結論を生成する（LLMの回答から「結論：〜」の1行のみ抽出）
        
        処理の流れ：
        1. search()で関連文書を検索
        2. 検索結果を文脈としてLLMに渡す
        3. LLMが生成した回答から「結論：〜」の行だけ抽出
        
        Args:
            question: 質問文
            k: 検索結果から使用する文書数
            w_sem: Semantic検索の重み
            w_key: BM25検索の重み
            candidate_k: リランキング前の候補数
        
        Returns:
            「結論：〜」の形式の1行のみ（日本語）
        """
        # =========================================
        # ステップ1: 検索して関連文書を取得
        # =========================================
        # 検索 → 上位k件の文脈をLLMへ
        # search()メソッドでハイブリッド検索 + リランキングを実行
        top_docs = self.search(question, k, w_sem, w_key, candidate_k)
        
        # 検索結果の文書を1つの文字列に結合
        # 各文書の内容を改行2つで区切って連結
        # 例：文書1の内容\n\n文書2の内容\n\n文書3の内容
        context = "\n\n".join(d.page_content for d in top_docs)

        # =========================================
        # ステップ2: プロンプトを生成
        # =========================================
        # PromptTemplateを使ってプロンプトを生成
        # {context} と {question} の部分が実際の値に置き換えられる
        prompt = self.prompt_template.format(context=context, question=question)
        
        # =========================================
        # ステップ3: LLMに質問して回答を生成
        # =========================================
        # LLMにプロンプトを渡して回答を生成
        # invoke() = LLMを呼び出して回答を取得
        raw_answer = self.llm.invoke(prompt)

        # =========================================
        # ステップ4: 回答から結論部分を抽出
        # =========================================
        # 結論1行だけ抽出
        # LLMの回答は複数行になることがあるので、「結論：」で始まる行を探す
        for line in raw_answer.split("\n"):
            if line.strip().startswith("結論"):
                return line.strip()

        # 保険：1行抽出できなければ先頭1行だけ返す
        # 「結論：」で始まる行が見つからなかった場合のフォールバック
        return raw_answer.split("\n")[0].strip()

    def generate_conclusion(self, question: str, k: int, w_sem: float, w_key: float, candidate_k: int = 60) -> str:
        """
        answer()のエイリアス（後方互換性のため）
        """
        return self.answer(question, k, w_sem, w_key, candidate_k)

# =========================================
# ③ 実行フェーズ
# =========================================
if __name__ == "__main__":
    # PDFファイルのパス（プロジェクトルートにあるPDFファイル）
    pdf_path = "linuxtext_ver4.0.0.pdf"

    # =========================================
    # フェーズ1: PDFの準備（インポート）
    # =========================================
    # ① PDFの仕込み（チャンク/embedding/キーワード検索辞書/ベクトルDB生成）
    # この処理で、PDFが検索可能な形式に変換される
    # 
    # 処理内容：
    # 1. PDFをテキストに変換
    # 2. テキストをチャンクに分割
    # 3. キーワード検索用のBM25辞書を作成
    # 4. セマンティック検索用のベクトルDBを作成
    importer = PDFRagSystem(persist_dir="./chroma_db")
    importer.import_pdf(pdf_path)
    # 注意：この処理は時間がかかることがあります（PDFのサイズによる）

    # =========================================
    # フェーズ2: RAGシステムの構築
    # =========================================
    # ② リランキングRAGを構築（検索/結論生成担当）
    # 検索と回答生成のためのシステムを初期化
    # 
    # 処理内容：
    # 1. セマンティック検索器の初期化
    # 2. キーワード検索器（BM25）の初期化
    # 3. リランキング用のCrossEncoderの初期化
    # 4. プロンプトテンプレートの初期化
    rerag = ReRankingRAG(importer.docs, persist_dir="./chroma_db")

    # =========================================
    # フェーズ3: リランキング動作の確認（デバッグ用）
    # =========================================
    # ②-1 リランキング動作の観察ログ（1回だけ表示）
    # リランキングが実際に順番を変えていることを確認するため、
    # テスト用の質問でリランキング前後の上位3件のIDを表示
    # 
    # この処理は、リランキングが正しく動作しているかを確認するためのものです
    test_question = "Linux 初心者 学習 実務 重要ポイント"
    
    # リランキング前の候補を取得（EnsembleRetrieverの結果）
    # ハイブリッド検索の結果を取得（リランキング前）
    ensemble = EnsembleRetriever(
        retrievers=[rerag.semantic, rerag.bm25],  # 2つの検索器
        weights=[0.5, 0.5]  # テスト用のweights（両方を均等に重視）
    )
    candidates_before = ensemble.get_relevant_documents(test_question)
    
    # リランキング後の候補を取得（CrossEncoderで再評価）
    # 質問と各候補のペアを作成
    pairs: List[Tuple[str, str]] = [(test_question, d.page_content) for d in candidates_before]
    # 各ペアの関連度スコアを計算
    scores = rerag.reranker.predict(pairs)
    # スコアの高い順に並び替え
    ranked = sorted(zip(candidates_before, scores), key=lambda x: x[1], reverse=True)
    candidates_after = [doc for doc, _score in ranked]
    
    # 上位3件のIDを表示（metadataからchunk_idを取得）
    # ヘルパー関数：Documentからchunk_idを取得
    def get_chunk_id(doc: Document) -> str:
        """Documentからchunk_idを取得する（なければフォールバック）"""
        return doc.metadata.get('chunk_id', f"page_{doc.metadata.get('page', 'unknown')}")
    
    # リランキング前後の上位3件のIDを取得
    before_ids = [get_chunk_id(doc) for doc in candidates_before[:3]]
    after_ids = [get_chunk_id(doc) for doc in candidates_after[:3]]
    
    # 結果を表示
    print("Rerank前 top3 IDs:", before_ids)
    print("Rerank後 top3 IDs:", after_ids)
    print()  # 空行を追加
    # 注意：リランキング前後でIDの順番が変わっていれば、リランキングが機能している証拠

    # ③ weights比較実験
    # Semantic検索とBM25検索の比重を5段階で変更し、
    # それぞれの比重でLLMが生成する結論がどう変わるかを観察する
    # 
    # weightsの意味：
    # - (0.9, 0.1): Semantic検索を重視（意味の類似性を優先）
    # - (0.7, 0.3): Semantic検索をやや重視
    # - (0.5, 0.5): 両方を均等に重視
    # - (0.3, 0.7): BM25検索をやや重視（キーワードマッチを優先）
    # - (0.1, 0.9): BM25検索を重視（キーワードマッチを優先）
    
    weight_cases = [
        (0.9, 0.1),  # Semantic: 90%, BM25: 10%
        (0.7, 0.3),  # Semantic: 70%, BM25: 30%
        (0.5, 0.5),  # Semantic: 50%, BM25: 50%
        (0.3, 0.7),  # Semantic: 30%, BM25: 70%
        (0.1, 0.9),  # Semantic: 10%, BM25: 90%
    ]

    # 専門用語が偏った質問（キーワード検索とセマンティック検索の差を観察するため）
    # 特定の技術用語や専門的な表現を多く含む質問にすることで、
    # BM25（キーワード検索）とセマンティック検索の挙動の違いが明確になる
    question = """
systemdのユニットファイルにおける依存関係の定義方法と、
systemctlコマンドによるサービス管理、journalctlによるログ確認、
シェルスクリプトのバックグラウンドプロセス管理におけるジョブコントロールとシグナルハンドリング、
iptablesのNATテーブルとフィルターテーブルにおけるパケット転送ルールの優先順位設定について、
結論のみを1行で日本語で述べてください。
""".strip()

    # 各weightsで検索→リランキング→LLM生成を実行
    # キーワード検索とセマンティック検索の差を観察するため、検索結果の比較も表示
    for w_sem, w_key in weight_cases:
        print(f"\n{'='*80}")
        print(f"【検索重み】Semantic: {w_sem:.1f}, Keyword: {w_key:.1f}")
        print(f"{'='*80}")
        
        # =========================================
        # 検索結果の比較表示（デバッグ用）
        # =========================================
        # キーワード検索（BM25）のみの結果（上位3件）
        # BM25 = キーワードマッチングに基づく検索
        # 質問に含まれる単語が多く出現する文書を優先的に返す
        bm25_results = rerag.bm25.get_relevant_documents(question)[:3]
        print("\n【キーワード検索（BM25）のみ - 上位3件】")
        for i, doc in enumerate(bm25_results, 1):
            chunk_id = doc.metadata.get('chunk_id', f"page_{doc.metadata.get('page', 'unknown')}")
            preview = doc.page_content[:100].replace('\n', ' ') + "..."  # 最初の100文字だけ表示
            print(f"  {i}. [{chunk_id}] {preview}")
        
        # セマンティック検索のみの結果（上位3件）
        # セマンティック検索 = 文章の意味を理解して検索
        # 質問の意味に近い文書を優先的に返す（キーワードが一致しなくてもOK）
        semantic_results = rerag.semantic.get_relevant_documents(question)[:3]
        print("\n【セマンティック検索のみ - 上位3件】")
        for i, doc in enumerate(semantic_results, 1):
            chunk_id = doc.metadata.get('chunk_id', f"page_{doc.metadata.get('page', 'unknown')}")
            preview = doc.page_content[:100].replace('\n', ' ') + "..."
            print(f"  {i}. [{chunk_id}] {preview}")
        
        # ハイブリッド検索の結果（リランキング前）を表示
        # ハイブリッド検索 = キーワード検索とセマンティック検索を組み合わせた検索
        # 重み（weights）によって、どちらの検索を重視するかが決まる
        ensemble = EnsembleRetriever(
            retrievers=[rerag.semantic, rerag.bm25],  # 2つの検索器
            weights=[w_sem, w_key]                   # それぞれの重み
        )
        hybrid_before = ensemble.get_relevant_documents(question)[:5]
        print("\n【ハイブリッド検索（リランキング前） - 上位5件】")
        print("※重みの違いが検索結果に反映されているか確認できます")
        for i, doc in enumerate(hybrid_before, 1):
            chunk_id = doc.metadata.get('chunk_id', f"page_{doc.metadata.get('page', 'unknown')}")
            preview = doc.page_content[:100].replace('\n', ' ') + "..."
            print(f"  {i}. [{chunk_id}] {preview}")
        
        # ハイブリッド検索 + リランキング後の結果
        # リランキング = 検索結果の順位を再評価して並び替えること
        # CrossEncoderを使って、より正確な関連度を計算
        top_docs = rerag.search(question, k=5, w_sem=w_sem, w_key=w_key, candidate_k=60)
        print("\n【ハイブリッド検索（リランキング後） - 上位5件】")
        print("※リランキングによって順位が変わっているか確認できます")
        for i, doc in enumerate(top_docs, 1):
            chunk_id = doc.metadata.get('chunk_id', f"page_{doc.metadata.get('page', 'unknown')}")
            preview = doc.page_content[:100].replace('\n', ' ') + "..."
            print(f"  {i}. [{chunk_id}] {preview}")
        
        # =========================================
        # LLMによる回答生成
        # =========================================
        # LLM生成（PromptTemplateを使用）
        # 検索結果を文脈として、LLMに質問を投げかける
        context = "\n\n".join(d.page_content for d in top_docs)
        prompt = rerag.prompt_template.format(context=context, question=question)
        raw_answer = rerag.llm.invoke(prompt)
        
        # 回答から「結論：」で始まる行を抽出
        for line in raw_answer.split("\n"):
            if line.strip().startswith("結論"):
                result = line.strip()
                break
        else:
            # 「結論：」で始まる行が見つからなかった場合のフォールバック
            result = raw_answer.split("\n")[0].strip()
        
        print(f"\n【LLM生成結論】")
        print(result)
