import re
from pathlib import Path
import shutil

from cleantext import clean
from tqdm.auto import tqdm


def fix_parathesis(text: str, re_str=r"(?<=[([]) +| +(?=[)\]])"):
    """
    fix_parathesis - does the following:
                        input text "I like ( perhaps even love ) to eat beans."
                        output text "I like (perhaps even love) to eat beans."
    """
    fixed_text = re.sub(re_str, "", text)

    return fixed_text


def fix_punct_spaces(input_text: str):
    """
    fix_punct_spaces - replace spaces around punctuation with punctuation. For example, "hello , there" -> "hello, there"
    :input_text: str, required, input string to be corrected
    Returns
    fixed_text - str, corrected string
    """

    fix_spaces = re.compile(r"\s*([?!.,]+(?:\s+[?!.,;:]+)*)\s*")
    input_text = fix_spaces.sub(
        lambda x: "{} ".format(x.group(1).replace(" ", "")), input_text
    )
    input_text = input_text.replace(" ' ", "'")
    fixed_text = input_text.replace(' " ', '"')
    return fix_parathesis(fixed_text.strip())


def custom_clean(ugly_txt, lowercase=True):

    return clean(ugly_txt, lower=lowercase)


def collapse_directory(directory: str or Path, verbose=False, ignore_errors=True):
    """
    collapse_directory - given a directory, uses pathlib to find all files recursively, and move them into the directory path, removing all sub-folders.

    :directory: str or Path, required, directory to be collapsed
    :verbose: bool, optional, default False, if True, prints status updates
    :ignore_errors: bool, optional, default True, if True, ignores errors when clearing sub-folders
    """
    directory = Path(directory)
    for f in directory.rglob("*"):
        if f.is_file():
            shutil.move(str(f), str(directory))

    # count the number of files in the new top level directory
    num_files = len(list(directory.rglob("*")))
    if verbose:
        print(f"{num_files} files moved to {directory}")

    # remove empty sub-directories
    for sub_dir in directory.rglob("*"):
        if sub_dir.is_dir():
            shutil.rmtree(sub_dir, ignore_errors=ignore_errors)

    if verbose:
        print(f"{directory.resolve()} is now reset to top level directory")
