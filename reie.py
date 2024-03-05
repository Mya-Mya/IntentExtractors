from iie import IIE
from intents import Intent,ETAGapQuery,HeightQuery
from re import compile

class ReIE(IIE):
    def __init__(self) -> None:
        self.heightquery_pat = compile(
            r"(lower|raise) the (min|max) height of (ip1|ip2) on the flight (\d*)"
        )
        self.etagapquery_pat = compile(
            r"(increase|decrease) the gap between the flight (\d*) and the (next|previous) one"
        )
    def extract(self, prompt: str) -> Intent:
        m = self.heightquery_pat.match(prompt)
        if m:
            op_r, bo_r, ip_r, re_r = m.groups()
            
            op = {"lower":"lower","raise":"raise"}[op_r]
            bo = {"min":"min","max":"max"}[bo_r]
            ip = {"ip1":"1","ip2":"2"}[ip_r]
            re = int(re_r)

            return Intent(
                queryType="height",
                content=HeightQuery(
                    reference=re,
                    operation=op,
                    boundary=bo,
                    ip=ip
                )
            )
        
        m = self.etagapquery_pat.match(prompt)
        if m:
            op_r, re_r, co_r = m.groups()

            op = {"increase":"increase","decrease":"decrease"}[op_r]
            re = int(re_r) - {"next":0,"previous":1}[co_r]
            
            return Intent(
                queryType="etagap",
                content=ETAGapQuery(
                    reference=re,
                    operation=op
                )
            )
        
        raise ValueError(f"Did not understand the {prompt=}")