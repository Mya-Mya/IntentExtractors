from typing import List, Tuple, Literal
import numpy as np
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise
from torch import no_grad
import matplotlib.pyplot as plt


@dataclass
class QAResult:
    top_preds: List[Tuple[float, int, int]]
    context_tokens: List[str]
    context_ids: List[int]
    s_logits: np.ndarray
    e_logits: np.ndarray


class QABot:
    def __init__(self, qamodel: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        self.model = qamodel
        self.tokenizer = tokenizer

    def execute(self, context: str, question: str, rankingmethod: Literal["separate", "together"] = "separate"):
        """
        QAタスクを実行する．`qa_tokenizer`と`qa_model`を使用する．

        Parameters
        ----------
        context : str
            本文．
        question : str
            質問文．
        rankingmethod : Literal["separate", "together"]
            回答の並べ方．
            "separate"では，始点ロジットと終点ロジットを別々に処理する．
            始点ロジットの最も大きいものと，終点ロジットの最も大きいものをそれぞれ始点，終点とする．
            したがって，回答は1つのみ返却される．
            "together"では，始点ロジットと終点ロジットの和が大きい順に，始点と終点を並べる．

        Returns
        -------
        r : QAResult
            QAタスクの実行結果．
        """
        c_tokenized = self.tokenizer(context)
        c_ids = c_tokenized.input_ids[1:-1]
        n_c_tokens = len(c_ids)
        c_tokens = [self.tokenizer.ids_to_tokens[i] for i in c_ids]

        with no_grad():
            outputs = self.model(
                **self.tokenizer(question, context, return_tensors="pt")
            )
        s_logits = outputs.start_logits[0, -n_c_tokens-1:-1].numpy()
        e_logits = outputs.end_logits[0, -n_c_tokens-1:-1].numpy()

        if rankingmethod == "together":
            preds = []
            for s, s_logit in enumerate(s_logits):
                for e, e_logit in enumerate(e_logits[s+1:], s+1):
                    logit = s_logit+e_logit
                    preds.append((logit, s, e))
            top_preds = sorted(
                preds, key=lambda pred: pred[0], reverse=True
            )
        elif rankingmethod == "separate":
            top_preds = [
                (-1, s_logits.argmax(), e_logits.argmax())
            ]
        else:
            raise ValueError(f"Illegal ranking method: {rankingmethod}")

        return QAResult(
            top_preds=top_preds,
            context_tokens=c_tokens,
            context_ids=c_ids,
            s_logits=s_logits,
            e_logits=e_logits
        )


def show_qaresult(ax:plt.Axes, r: QAResult):
    ax.get_figure().set_figwidth(10)
    ax.plot(r.s_logits, marker="o", ls=":", label="Start")
    ax.plot(r.e_logits, marker="s", ls=":", label="End")
    ax.set_xlabel("Token")
    ax.set_ylabel("Logit")
    ax.grid(axis="x")
    ax.set_xticks(np.arange(len(r.context_tokens)), r.context_tokens)
    for label in ax.get_xticklabels():
        label.set_rotation(40)
        label.set_horizontalalignment('right')
    ax.legend()
    plt.show()


class Detector:
    def __init__(self, st: SentenceTransformer) -> None:
        self.model = st
        self.sentence_to_e = {}

    def encode_cache(self, sentence: str):
        if not sentence in self.sentence_to_e:
            self.sentence_to_e[sentence] = self.model.encode(sentence)
        return self.sentence_to_e[sentence]

    def get_closet_proxy(self, proxies: List[str], x: str):
        """
        与えられた文`x`が代表文`proxies`のうちどれに近いかを判定する．`sb_model`を使用する．
        Parameters
        ----------
        proxies : List[str]
            代表文．
        x : str
            対象となる文．

        Returns
        -------
        i : int
            `x`が最も近い代表文のインデックス．
        """
        proxy_e_s = list(map(self.encode_cache, proxies))
        x_e = self.encode_cache(x)
        similarity_result = pairwise.cosine_similarity(proxy_e_s, [x_e])
        i = similarity_result.argmax()
        return i
