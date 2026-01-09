"""
単一ソース対応ReRankingRAGシステム

ハイブリッド検索（Semantic + Keyword）とリランキングを使用したRAGシステム。
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder
from typing import List, Tuple

from ..utils.text import clean_text


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
        """
        初期化メソッド
        
        Args:
            docs: 検索対象のチャンク化された文書リスト
            persist_dir: ベクトルデータベースの保存フォルダのパス
        """
        # 検索対象チャンクを保持
        self.docs = docs
        self.persist_dir = persist_dir

        # embeddingモデル（Semantic検索用、384次元で統一）
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # LLMモデル（Ollama llama3）
        self.llm = Ollama(model="llama3:latest", temperature=0.0)

        # セマンティック検索器の初期化
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        self.semantic = self.vectorstore.as_retriever(search_kwargs={"k": 60})

        # キーワード検索器（BM25）の初期化
        self.bm25 = BM25Retriever.from_documents(self.docs)
        self.bm25.k = 60

        # リランキング用のCrossEncoderの初期化
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # プロンプトテンプレートの初期化（デフォルト値、後で上書き可能）
        template = """【参考情報】
{context}

【質問】
{question}

【指示】
- 必ず日本語で回答する
- 文脈に書かれている事実のみを使用する
- 推測や一般知識を混ぜない
- 答えられない場合は正直にその旨を伝える
- 余計な説明はせず「結論：〜」の1行だけ出力してください"""
        
        self.prompt_template = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

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
        # ハイブリッド検索（Semantic + Keyword）
        ensemble = EnsembleRetriever(
            retrievers=[self.semantic, self.bm25],
            weights=[w_sem, w_key]
        )
        candidates = ensemble.get_relevant_documents(question)

        # リランキング（検索結果の再評価）
        pairs: List[Tuple[str, str]] = [(question, d.page_content) for d in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        # 上位k件を返す
        return [doc for doc, _score in ranked[:k]]

    def answer(self, question: str, k: int, w_sem: float, w_key: float, candidate_k: int = 60) -> str:
        """
        質問に対する回答を生成する（LLMの回答から「結論：〜」の1行のみ抽出）
        
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
        # プロンプトテンプレートが設定されていない場合はエラー
        if self.prompt_template is None:
            raise ValueError("プロンプトテンプレートが設定されていません。")
        
        # 検索 → 上位k件の文脈をLLMへ
        top_docs = self.search(question, k, w_sem, w_key, candidate_k)
        
        # 検索結果が空の場合はエラーメッセージを返す
        if not top_docs:
            return "申し訳ございませんが、関連する情報が見つかりませんでした。"
        
        # コンテキストを構築
        context = "\n\n".join(d.page_content for d in top_docs)

        # プロンプトを生成
        prompt = self.prompt_template.format(context=context, question=question)
        
        # LLMに質問して回答を生成
        raw_answer = self.llm.invoke(prompt)

        # 結論1行だけ抽出
        for line in raw_answer.split("\n"):
            if line.strip().startswith("結論"):
                return line.strip()
        
        # 保険：1行抽出できなければ先頭1行だけ返す
        return raw_answer.split("\n")[0].strip()

    def generate_conclusion(self, question: str, k: int, w_sem: float, w_key: float, candidate_k: int = 60) -> str:
        """
        answer()のエイリアス（後方互換性のため）
        """
        return self.answer(question, k, w_sem, w_key, candidate_k)
