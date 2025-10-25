# -*- coding: utf-8 -*-
import re
import jieba
from pypinyin import pinyin, Style

# --------------------- 初始化映射表 ---------------------
# 声母点位映射（根据标准第5章）
INITIALS_MAP = {
    'b': {1, 2},
    'p': {1, 2, 3, 4},
    'm': {1, 3, 4},
    'f': {1, 2, 4},
    'd': {1, 4, 5},
    't': {2, 3, 4, 5},
    'n': {1, 3, 4, 5},
    'l': {1, 2, 3},
    'g': {1, 2, 4, 5},
    'j': {1, 2, 4, 5},  # g/j 同形
    'k': {1, 3},
    'q': {1, 3},  # k/q 同形
    'h': {1, 2, 5},
    'x': {1, 2, 5},  # h/x 同形
    'zh': {3, 4},
    'ch': {1, 2, 3, 4, 5},
    'sh': {1, 5, 6},
    'r': {2, 4, 5},
    'z': {3, 4},
    'c': {1, 4},
    's': {2, 3, 4},
    'y': {1,3},    # y的独立声母符号⠽（第4、5点）
    'w': {2,4,5,6}, # w的独立声母符号⠺（第2、4、5、6点）

}

# 韵母点位映射（根据标准第6章）
FINALS_MAP = {
    'a': {3, 5},
    'o': {2, 6},
    'e': {2, 6},
    'i': {2, 4},
    'u': {1, 3, 6},
    'ü': {3, 4, 6},
    'er': {1, 2, 3, 5},
    'ai': {2, 4, 6},
    'ei': {2, 3, 4, 6},
    'ao': {2, 3, 5},
    'ou': {1, 2, 3, 5, 6},
    'ia': {1, 2, 4, 6},
    'iao': {3, 4, 5},
    'ie': {1, 5},
    'iu': {1, 2, 4, 5, 6},
    'ian': {1, 2, 4, 6},
    'in': {1, 2, 6},
    'iang': {1, 3, 4, 6},
    'ing': {1, 6},
    'uang': {2, 3, 5, 6},
    'ong': {2, 5, 6},
    'ua': {1, 3, 4, 5, 6},
    'uai': {1, 3, 4, 5, 6},
    'ui': {1, 2, 5, 6},
    'ue': {1, 2, 3, 4, 5, 6},
    'iong': {1, 4, 5, 6},
    'an': {1, 2, 3, 6},
    'ang': {2, 3, 6},
    'en': {3, 5, 6},
    'eng': {3, 4, 5, 6},
    'un': {4, 5, 6},
    'ong': {2, 5, 6},
    'ueng': {1, 2, 3, 4, 5, 6},
    'uan': {1, 2, 4, 5, 6},
    'uang': {2, 3, 5, 6},
    'uei': {1, 2, 3, 4, 5, 6},
    'uen': {2, 5, 6},
    'u': {1, 3, 6},
    'üe': {1, 2, 3, 4, 5, 6},
    'ün': {1, 2, 3, 4, 5, 6}
}

# 声调符号（标准第7章）
TONES_MAP = {
    1: set(),       # 阴平（通常不标）
    2: {3},         # 阳平
    3: {3, 6},      # 上声
    4: {6},         # 去声
    5: set()        # 轻声
}

# 简写规则（标准第11章）
CONTRACTIONS = {
    '的': {2, 4, 5},
    '么': {3, 4},
    '你': {1, 3, 4, 5},
    '他': {1, 2, 3, 5},
    '她': {1, 2, 3, 5},
    '它': {1, 2, 3, 5}
}

# 标点符号（标准第8章）
PUNCTUATION_MAP = {
    '。': {2, 5, 6},    # 句号 ⠲
    '，': {2},          # 逗号 ⠂
    '、': {2, 6},       # 顿号 ⠄
    '？': {2, 3, 6},    # 问号 ⠦
    '！': {2, 3, 5},    # 叹号 ⠖
    '：': {2, 5},       # 冒号 ⠒
    '“': {2, 3, 5, 6},  # 前双引号 ⠶
    '”': {3, 5, 6},     # 后双引号 ⠴
    '（': {5, 6, 3},    # 左括号 ⠷
    '）': {5, 6, 2},    # 右括号 ⠾
    '《': {5, 6, 3},    # 左书名号 ⠷
    '》': {5, 6, 2},    # 右书名号 ⠾
    '；': {5, 6},       # 分号 ⠶
    '‘': {5, 6, 3},     # 左单引号 ⠷
    '’': {5, 6, 2},     # 右单引号 ⠾
    '…': {5, 5, 5}      # 省略号 ⠵⠵⠵
}

# 数字前缀符号（附录B）
NUM_PREFIX = {3, 4, 5, 6}

# 数字映射（附录B）
NUM_MAP = {
    '0': {3, 5, 6},
    '1': {1},
    '2': {1, 2},
    '3': {1, 4},
    '4': {1, 4, 5},
    '5': {1, 5},
    '6': {1, 2, 4},
    '7': {1, 2, 4, 5},
    '8': {1, 2, 5},
    '9': {2, 4}
}

# --------------------- 核心转换逻辑 ---------------------
class BrailleConverter:
    def __init__(self):
        jieba.initialize()
        
    def convert(self, text):
        """主转换方法"""
        # 预处理：分离中英文/数字
        segments = self._segment_text(text)
        
        braille = []
        for seg in segments:
            if seg['type'] == 'zh':
                words = list(jieba.cut(seg['content']))
                for word in words:
                    braille.append(self._convert_word(word))
            elif seg['type'] == 'num':
                braille.append(self._convert_number(seg['content']))
            elif seg['type'] == 'en':
                braille.append(self._convert_english(seg['content']))
            else:  # 标点
                braille.append(self._get_punct(seg['content']))
                
        return ''.join(braille)
    
    def _segment_text(self, text):
        """分离中英数混排文本"""
        segments = []
        for part in re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+|[^\w\s]', text):
            if re.match(r'[\u4e00-\u9fff]', part):
                segments.append({'type': 'zh', 'content': part})
            elif part.isdigit():
                segments.append({'type': 'num', 'content': part})
            elif part.isalpha():
                segments.append({'type': 'en', 'content': part})
            else:
                segments.append({'type': 'punct', 'content': part})
        return segments
    
    def _convert_word(self, word):
        """转换汉字为盲文（支持y/w独立声母）"""
        # 简写优先处理
        if word in CONTRACTIONS:
            return self._points_to_char(CONTRACTIONS[word])

        chars = []
        for char in word:
            # 获取拼音及声调
            pinyin_str = pinyin(char, style=Style.TONE3)[0][0]
            tone = int(pinyin_str[-1]) if pinyin_str[-1].isdigit() else 5
            pinyin_part = pinyin_str[:-1] if pinyin_str[-1].isdigit() else pinyin_str

            # 分解声母韵母
            initial, final = self._split_initial_final(pinyin_part)

            # ---- 新增逻辑：处理y/w的隔音符号 ----
            if initial in ['y', 'w']:
                # 添加隔音符号⠠（第4、5、6点）
                chars.append(self._points_to_char({4,5,6}))
                # 添加y/w的声母符号
                chars.append(self._points_to_char(INITIALS_MAP[initial]))
            elif initial:  # 普通声母
                chars.append(self._points_to_char(INITIALS_MAP.get(initial, set())))

            # 处理韵母
            final_pts = FINALS_MAP.get(final, set())
            chars.append(self._points_to_char(final_pts))



        return ''.join(chars)
    
    def _split_initial_final(self, pinyin_part):
        """分离声母和韵母（确保y/w优先匹配）"""
        # 调整声母列表顺序，将y/w提前到普通声母前
        initials = [
            'zh', 'ch', 'sh',
            'z', 'c', 's', 'r',
            'b', 'p', 'm', 'f',
            'd', 't', 'n', 'l',
            'g', 'k', 'h',
            'j', 'q', 'x',
            'y', 'w',  # 新增y/w作为独立声母
        ]
        
        for initial in initials:
            if pinyin_part.startswith(initial):
                return initial, pinyin_part[len(initial):]
        return '', pinyin_part
    
    def _get_tone_points(self, initial, tone):
        """获取声调点位（应用省写规则）"""
        # 声调省写规则（标准第10章）
        if tone == 1 and initial == 'f':
            return set()  # f声母省写阴平
        if tone == 2 and initial in {'p', 'm', 't', 'n', 'h', 'q', 'ch', 'r', 'c'}:
            return set()  # 阳平省写
        if tone == 4 and initial in {'b', 'd', 'l', 'g', 'k', 'j', 'x', 'zh', 'sh', 'z', 's'}:
            return set()  # 去声省写
        return TONES_MAP.get(tone, set())
    
    def _convert_number(self, num_str):
        """转换数字为盲文"""
        braille = [self._points_to_char(NUM_PREFIX)]  # 数字前缀
        for n in num_str:
            braille.append(self._points_to_char(NUM_MAP[n]))
        return ''.join(braille)
    
    def _convert_english(self, eng_str):
        """转换英文为盲文（简化版）"""
        braille = []
        for char in eng_str.lower():
            # 英文字母映射（简化版）
            if char == 'a':
                braille.append(self._points_to_char({1}))
            elif char == 'b':
                braille.append(self._points_to_char({1, 2}))
            elif char == 'c':
                braille.append(self._points_to_char({1, 4}))
            elif char == 'd':
                braille.append(self._points_to_char({1, 4, 5}))
            elif char == 'e':
                braille.append(self._points_to_char({1, 5}))
            elif char == 'f':
                braille.append(self._points_to_char({1, 2, 4}))
            elif char == 'g':
                braille.append(self._points_to_char({1, 2, 4, 5}))
            elif char == 'h':
                braille.append(self._points_to_char({1, 2, 5}))
            elif char == 'i':
                braille.append(self._points_to_char({2, 4}))
            elif char == 'j':
                braille.append(self._points_to_char({2, 4, 5}))
            elif char == 'k':
                braille.append(self._points_to_char({1, 3}))
            elif char == 'l':
                braille.append(self._points_to_char({1, 2, 3}))
            elif char == 'm':
                braille.append(self._points_to_char({1, 3, 4}))
            elif char == 'n':
                braille.append(self._points_to_char({1, 3, 4, 5}))
            elif char == 'o':
                braille.append(self._points_to_char({1, 3, 5}))
            elif char == 'p':
                braille.append(self._points_to_char({1, 2, 3, 4}))
            elif char == 'q':
                braille.append(self._points_to_char({1, 2, 3, 4, 5}))
            elif char == 'r':
                braille.append(self._points_to_char({1, 2, 3, 5}))
            elif char == 's':
                braille.append(self._points_to_char({2, 3, 4}))
            elif char == 't':
                braille.append(self._points_to_char({2, 3, 4, 5}))
            elif char == 'u':
                braille.append(self._points_to_char({1, 3, 6}))
            elif char == 'v':
                braille.append(self._points_to_char({1, 2, 3, 6}))
            elif char == 'w':
                braille.append(self._points_to_char({2, 4, 5, 6}))
            elif char == 'x':
                braille.append(self._points_to_char({1, 3, 4, 6}))
            elif char == 'y':
                braille.append(self._points_to_char({1, 3, 4, 5, 6}))
            elif char == 'z':
                braille.append(self._points_to_char({1, 3, 5, 6}))
        return ''.join(braille)
    
    def _get_punct(self, char):
        """转换标点符号为盲文"""
        points = PUNCTUATION_MAP.get(char, set())
        return self._points_to_char(points)
    
    def _points_to_char(self, points):
        """点位转Unicode字符"""
        code = 0x2800
        for p in points:
            if 1 <= p <= 6:
                code |= 1 << (p - 1)
        return chr(code)
    
    def get_braille_sets_for_word(self, text):
        """
        将单个词语转换为盲文点位集合的列表。
        每个声母和韵母都作为独立的盲文字符。
        e.g., '你好' -> [{1, 3, 4, 5}, {1, 2, 5}, {2, 3, 5}] 
        (你(简写), h, ao)
        """
        braille_sets = []
        # 假设一个词语不包含复杂的中英数混合
        words = list(jieba.cut(text))
        for word in words:
            # 简写优先处理
            if word in CONTRACTIONS:
                braille_sets.append(CONTRACTIONS[word])
                continue

            # 正常拼写处理
            pinyin_list = pinyin(word, style=Style.TONE3)
            for p in pinyin_list:
                pinyin_str = p[0]
                initial, final = self._split_initial_final(pinyin_str.rstrip('12345'))
                
                # 添加声母点位
                if initial:
                    initial_points = INITIALS_MAP.get(initial)
                    if initial_points:
                        braille_sets.append(initial_points)
                
                # 添加韵母点位
                if final:
                    final_points = FINALS_MAP.get(final)
                    if final_points:
                        braille_sets.append(final_points)

        return braille_sets

    def get_braille_dots_list(self, text):
        """将文本转换为扁平的盲文点位数字列表，用于音频合成。"""
        all_dots = []
        braille_sets = self.get_braille_sets_for_word(text)
        for s in braille_sets:
            all_dots.extend(sorted(list(s))) # 排序以保证音频顺序
        return all_dots

    def get_braille_info(self, text):
        """将文本转换为盲文并返回详细信息"""
        braille_output = self.convert(text)
        
        # 生成点位信息
        dot_info = []
        for char in braille_output:
            code = ord(char)
            points = []
            for i in range(8):
                if (code >> i) & 1:
                    points.append(str(i + 1))
            dot_info.append(f"{char}: 点位 {','.join(points)}")
            
        return f"原文: {text}\n盲文: {braille_output}\n点位信息: \n" + "\n".join(dot_info)

# --------------------- 使用示例 ---------------------
if __name__ == "__main__":
    converter = BrailleConverter()
    
    # 测试文本
    text = "你好"
    braille = converter.convert(text)
    
    # 输出结果
    print(f"原文: {text}")
    print(f"盲文: {braille}")  # 需要盲文字体支持显示
    
    # 获取详细信息
    info = converter.get_braille_info(text)
    print(info)