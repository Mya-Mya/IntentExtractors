import json
from typing import Literal
from argparse import ArgumentParser
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
from iie import IIE
from intents import Intent, ETAGapQuery, HeightQuery
from utils import QAResult, QABot, Detector
from re import match

INSTRUCTION_DATA = {}


class NLPIE(IIE):
    def __init__(self, sbert_name: str, qamodel_name: str) -> None:
        global INSTRUCTION_DATA
        INSTRUCTION_DATA = json.load(
            Path("./nlpie_instructions.json").open("r"))
        self.sbert = SentenceTransformer(sbert_name)
        self.qamodel: DistilBertForQuestionAnswering = DistilBertForQuestionAnswering.from_pretrained(
            qamodel_name)
        self.qatokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained(
            qamodel_name)
        self.qabot = QABot(self.qamodel, self.qatokenizer)
        self.detector = Detector(self.sbert)

    def extract(self, prompt: str) -> Intent:
        """
        与えられたプロンプトをもとに，意図(Intent)を生成する．
        """
        type_i = self.detector.get_closet_proxy(
            proxies=[
                INSTRUCTION_DATA["QueryTypeDetectionProxies"]["ETAGap"],
                INSTRUCTION_DATA["QueryTypeDetectionProxies"]["Height"]
            ],
            x=prompt
        )
        method = [self.extract_with_etagap_query,
                  self.extract_with_height_query][type_i]
        intent = method(prompt)
        return intent

    def extract_with_etagap_query(self, prompt: str) -> Intent:
        """
        このメソッドでは，与えられたプロンプトが到着時間差の制約に関するクエリ(ETAGap Query)であるとし，
        意図を生成する．
        """
        # [re] Reference - Classification
        re_r = self.qabot.execute(
            context=prompt,
            question=INSTRUCTION_DATA["ETAGapQuery"]["ReferenceQuestion"]
        )
        re_pred = re_r.top_preds[0]
        re_command = self.qatokenizer.decode(
            re_r.context_ids[re_pred[1]:re_pred[2]+1])
        # Validate re_command with regex
        m = match(r"(\d) and (\d)", re_command)
        if m is None or len(m.groups()) == 0:
            raise ValueError(
                f"Could not extract data from {re_command=} : Does not match '¥d and ¥d'.")
        re = min(int(m.group(1)), int(m.group(2)))

        # [op] Operation - Detection
        op_i = self.detector.get_closet_proxy(
            proxies=[
                INSTRUCTION_DATA["ETAGapQuery"]["OperationProxies"]["Increase"],
                INSTRUCTION_DATA["ETAGapQuery"]["OperationProxies"]["Decrease"]
            ],
            x=prompt
        )
        op = ("increase", "decrease")[op_i]

        # Make Intent
        return Intent(
            queryType="etagap",
            content=ETAGapQuery(
                reference=re,
                operation=op
            )
        )

    def extract_with_height_query(self, prompt: str) -> Intent:
        """
        このメソッドでは，与えられたプロンプトが高度の制約に関するクエリ(Height Query)であるとし，
        意図を生成する．
        """
        # [re] Reference - Classification
        re_r = self.qabot.execute(
            context=prompt,
            question=INSTRUCTION_DATA["HeightQuery"]["ReferenceQuestion"]
        )
        re_pred = re_r.top_preds[0]
        re_command = self.qatokenizer.decode(
            re_r.context_ids[re_pred[1]:re_pred[2]+1])
        # Validate re_command with regex
        m = match(r"(\d*)", re_command)
        if m is None or len(m.groups()) == 0:
            raise ValueError(
                f"Could not extract data from {re_command=} : Does not match '¥d*'")
        re = int(m.groups()[0])

        # [op] Operation - Detection
        op_i = self.detector.get_closet_proxy(
            proxies=[
                INSTRUCTION_DATA["HeightQuery"]["OperationProxies"]["Lower"],
                INSTRUCTION_DATA["HeightQuery"]["OperationProxies"]["Raise"]
            ],
            x=prompt
        )
        op = ("lower", "raise")[op_i]

        # [bo] Boundary - Detection
        bo_i = self.detector.get_closet_proxy(
            proxies=[
                INSTRUCTION_DATA["HeightQuery"]["BoundaryProxies"]["Min"],
                INSTRUCTION_DATA["HeightQuery"]["BoundaryProxies"]["Max"]
            ],
            x=prompt
        )
        bo = ("min", "max")[bo_i]

        # [ip] IP - Detection
        """ip_i = self.detector.get_closet_proxy(
            proxies=[
                INSTRUCTION_DATA["HeightQuery"]["IPProxies"]["IP1"],
                INSTRUCTION_DATA["HeightQuery"]["IPProxies"]["IP2"]
            ],
            x=prompt
        )
        ip = ("1", "2")[ip_i]"""

        # [ip] IP - Classification
        ip_r = self.qabot.execute(
            context=prompt,
            question=INSTRUCTION_DATA["HeightQuery"]["IPQuestion"]
        )
        ip_pred = ip_r.top_preds[0]
        ip_command = self.qatokenizer.decode(
            ip_r.context_ids[ip_pred[1]:ip_pred[2]+1])
        # Validate re_command with regex
        m = match(r"(.*)(1|2)", ip_command)
        if m is None or len(m.groups()) == 0:
            raise ValueError(
                f"Could not extract data from {ip_command=} : Does not match '(ip)?\s?(1|2)'")
        ip = m.groups()[1]

        # Make Intent
        return Intent(
            queryType="height",
            content=HeightQuery(
                reference=re,
                operation=op,
                boundary=bo,
                ip=ip
            )
        )