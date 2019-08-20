import re

REMAP = {"-LRB-": "(", "-RRB-": ")", "-LCB-": "{", "-RCB-": "}",
         "-LSB-": "[", "-RSB-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-LRB-|-RRB-|-LCB-|-RCB-|-LSB-|-RSB-|``|''",
        lambda m: REMAP.get(m.group()), x)


MAP = {" ,": ",", " ?": "?", " !": "!", " .": ".", "( ": "(", " )": ")",
       "{ ": "{", " }": "}", "[ ": "[", " ]": "]"}


def tidy(text):

    for token in MAP:
        text = text.replace(token, MAP[token])

    return text
