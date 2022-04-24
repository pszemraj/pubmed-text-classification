import re
from pathlib import Path

from cleantext import clean
from tqdm.auto import tqdm


def fix_parathesis(text:str,
                   re_str=r"(?<=[([]) +| +(?=[)\]])"):
    """
    fix_parathesis - does the following:
                        input text "I like ( perhaps even love ) to eat beans."
                        output text "I like (perhaps even love) to eat beans."
    """
    fixed_text = re.sub(re_str, "", text)

    return fixed_text

def fix_punct_spaces(input_text:str):
    """
    fix_punct_spaces - replace spaces around punctuation with punctuation. For example, "hello , there" -> "hello, there"
    :input_text: str, required, input string to be corrected
    Returns
    fixed_text - str, corrected string
    """

    fix_spaces = re.compile(r"\s*([?!.,]+(?:\s+[?!.,;:]+)*)\s*")
    input_text = fix_spaces.sub(lambda x: "{} ".format(x.group(1).replace(" ", "")), input_text)
    input_text = input_text.replace(" ' ", "'")
    fixed_text = input_text.replace(' " ', '"')
    return fix_parathesis(fixed_text.strip())

def custom_clean(ugly_txt, lowercase=True):

    return clean(ugly_txt, lower=lowercase)
