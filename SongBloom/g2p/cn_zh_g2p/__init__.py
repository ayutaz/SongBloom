from . import chinese, english, japanese
from .symbols import *
import yaml
language_module_map = {"zh": chinese, "en": english, "ja": japanese}

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

import re

# def split_text(text):
#     chinese_pattern = r'[\u4e00-\u9fa5][\u4e00-\u9fa5\ \,\.\!\?\，\。]+'
#     english_pattern = r'[a-zA-Z][a-zA-Z\'\ \,\.\!\?]+'
    
#     chinese_text = re.findall(chinese_pattern, text)
#     print(chinese_text)
#     english_text = re.findall(english_pattern, text)
    
#     return chinese_text, english_text

def split_text(text):
    # Merge punctuation replacement maps from all languages
    all_rep_map = {**chinese.rep_map}
    if hasattr(japanese, 'rep_map'):
        all_rep_map.update(japanese.rep_map)

    pattern = re.compile("|".join(re.escape(p) for p in all_rep_map.keys()))
    text = pattern.sub(lambda x: all_rep_map[x.group()], text)

    result = []
    lang = []
    buffer = ""
    # Chinese characters (CJK Unified Ideographs)
    chinese_pattern = r'[\u4e00-\u9fa5]'
    # Japanese hiragana and katakana (unique to Japanese)
    japanese_pattern = r'[\u3040-\u309F\u30A0-\u30FF]'
    # CJK characters that could be Chinese or Japanese
    cjk_pattern = r'[\u4e00-\u9fa5\u3040-\u309F\u30A0-\u30FF]'
    special_pattern = r'[\,\.\!\?\…\-]'

    def detect_lang(text_segment):
        """Detect language of a text segment."""
        # If contains hiragana/katakana, it's Japanese
        if re.search(japanese_pattern, text_segment):
            return 'ja'
        # If contains Chinese characters only, it's Chinese
        if re.search(chinese_pattern, text_segment):
            return 'zh'
        # Otherwise it's English
        return 'en'

    for char in text:
        if re.match(special_pattern, char):
            if buffer:
                result.append(buffer)
                lang.append(detect_lang(buffer))
            result.append(char)
            lang.append('sp')
            buffer = ""
        elif re.match(cjk_pattern, char):
            # CJK character (Chinese or Japanese)
            if buffer and not re.match(cjk_pattern, buffer[-1]):
                result.append(buffer)
                lang.append(detect_lang(buffer))
                buffer = ""
            buffer += char
        else:
            # Non-CJK character (likely English/ASCII)
            if buffer and re.match(cjk_pattern, buffer[-1]):
                result.append(buffer)
                lang.append(detect_lang(buffer))
                buffer = ""
            buffer += char

    if buffer:
        result.append(buffer)
        lang.append(detect_lang(buffer))

    return result, lang

def mixed_language_to_phoneme(text):
    segments, lang = split_text(text)
    # print(segments, lang)
    result = [language_to_phoneme(s, l) for s, l in zip(segments, lang)]
    phones, word2ph = [], []
    for p, w, n in result:
        phones += p
        if w is None:
            w = []
        word2ph += w
    return phones, word2ph


def language_to_phoneme(text, language):
    if language == 'sp':
        return [text], None, text
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    if language == "zh":
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    else:
        try:
            phones = language_module.g2p(norm_text)
        except:
            phones = [norm_text]
        word2ph = None

    # for ph in phones:
    #     assert ph in symbols, ph
    return phones, word2ph, norm_text

def gen_vocabs():
    yaml.dump(symbols, open('./vocab.yaml', 'w'))

class G2P_Mix():
    def __call__(self, text):
        phones, word2ph = mixed_language_to_phoneme(text)
        return ' '.join(phones)