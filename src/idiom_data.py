#!/usr/bin/env python3
"""
成语数据加载模块
支持从多种来源加载成语数据，并提供首字/尾字索引
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
import json
import os


class IdiomDictionary:
    """
    成语字典类
    提供高效的首字/尾字索引，支持字面匹配和音同匹配
    """
    
    def __init__(self, use_pinyin: bool = False):
        """
        初始化成语字典
        
        Args:
            use_pinyin: 是否启用音同匹配（需要拼音数据）
        """
        self.use_pinyin = use_pinyin
        
        # 核心数据结构
        self.idioms: Dict[int, Tuple[str, Optional[str]]] = {}  # id -> (text, pinyin)
        self.id_to_text: Dict[int, str] = {}
        self.id_to_pinyin: Dict[int, Optional[str]] = {}
        
        # 索引结构
        self.head_index: Dict[str, List[int]] = defaultdict(list)  # 首字 -> 成语ID列表
        self.tail_index: Dict[str, List[int]] = defaultdict(list)  # 尾字 -> 成语ID列表
        
        # 音同索引（可选）
        self.head_pinyin_index: Dict[str, List[int]] = defaultdict(list)
        self.tail_pinyin_index: Dict[str, List[int]] = defaultdict(list)
        
        # 统计信息
        self.total_count = 0
        self.head_char_count: Dict[str, int] = defaultdict(int)
        self.tail_char_count: Dict[str, int] = defaultdict(int)
    
    def add_idiom(self, idiom_id: int, text: str, pinyin: Optional[str] = None) -> None:
        """
        添加单个成语
        
        Args:
            idiom_id: 成语唯一标识
            text: 成语文本（必须为4字）
            pinyin: 拼音（可选，格式如 "yi xin yi yi"）
        """
        if len(text) != 4:
            raise ValueError(f"成语必须是4字: {text}")
        
        self.idioms[idiom_id] = (text, pinyin)
        self.id_to_text[idiom_id] = text
        self.id_to_pinyin[idiom_id] = pinyin
        
        # 字面索引
        head_char = text[0]
        tail_char = text[-1]
        self.head_index[head_char].append(idiom_id)
        self.tail_index[tail_char].append(idiom_id)
        self.head_char_count[head_char] += 1
        self.tail_char_count[tail_char] += 1
        
        # 音同索引
        if self.use_pinyin and pinyin:
            head_pinyin = pinyin.split()[0] if pinyin.split() else ""
            tail_pinyin = pinyin.split()[-1] if pinyin.split() else ""
            if head_pinyin:
                self.head_pinyin_index[head_pinyin].append(idiom_id)
            if tail_pinyin:
                self.tail_pinyin_index[tail_pinyin].append(idiom_id)
        
        self.total_count += 1
    
    def load_from_list(self, idioms: List[Tuple[int, str, Optional[str]]]) -> None:
        """
        从列表批量加载成语
        
        Args:
            idioms: [(id, text, pinyin), ...]
        """
        for idiom_id, text, pinyin in idioms:
            self.add_idiom(idiom_id, text, pinyin)
    
    def load_from_json(self, filepath: str) -> None:
        """
        从JSON文件加载成语
        
        Args:
            filepath: JSON文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for i, item in enumerate(data):
            if isinstance(item, dict):
                text = item.get('word') or item.get('text') or item.get('idiom')
                pinyin = item.get('pinyin')
            elif isinstance(item, str):
                text = item
                pinyin = None
            else:
                continue
            
            if text and len(text) == 4:
                self.add_idiom(i, text, pinyin)
    
    def get_by_head(self, char: str, use_pinyin: bool = False) -> List[int]:
        """
        获取所有以某字开头的成语
        
        Args:
            char: 首字（或拼音）
            use_pinyin: 是否按拼音匹配
        
        Returns:
            成语ID列表
        """
        if use_pinyin and self.use_pinyin:
            return self.head_pinyin_index.get(char, [])
        return self.head_index.get(char, [])
    
    def get_by_tail(self, char: str, use_pinyin: bool = False) -> List[int]:
        """
        获取所有以某字结尾的成语
        
        Args:
            char: 尾字（或拼音）
            use_pinyin: 是否按拼音匹配
        
        Returns:
            成语ID列表
        """
        if use_pinyin and self.use_pinyin:
            return self.tail_pinyin_index.get(char, [])
        return self.tail_index.get(char, [])
    
    def get_followers(self, idiom_id: int, use_pinyin: bool = False) -> List[int]:
        """
        获取可以接在当前成语后的所有成语
        
        Args:
            idiom_id: 当前成语ID
            use_pinyin: 是否允许音同匹配
        
        Returns:
            可接龙的成语ID列表
        """
        text = self.id_to_text[idiom_id]
        tail_char = text[-1]
        
        # 字面匹配
        followers = set(self.head_index.get(tail_char, []))
        
        # 音同匹配（可选）
        if use_pinyin and self.use_pinyin:
            pinyin = self.id_to_pinyin[idiom_id]
            if pinyin:
                tail_pinyin = pinyin.split()[-1] if pinyin.split() else ""
                if tail_pinyin:
                    followers.update(self.head_pinyin_index.get(tail_pinyin, []))
        
        return list(followers)
    
    def get_text(self, idiom_id: int) -> str:
        """获取成语文本"""
        return self.id_to_text[idiom_id]
    
    def get_pinyin(self, idiom_id: int) -> Optional[str]:
        """获取成语拼音"""
        return self.id_to_pinyin[idiom_id]
    
    def get_all_ids(self) -> List[int]:
        """获取所有成语ID"""
        return list(self.idioms.keys())
    
    def get_tail_chars(self) -> Set[str]:
        """获取所有尾字集合"""
        return set(self.tail_index.keys())
    
    def get_head_chars(self) -> Set[str]:
        """获取所有首字集合"""
        return set(self.head_index.keys())
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计数据字典
        """
        # 计算高频尾字
        sorted_tail = sorted(self.tail_char_count.items(), key=lambda x: -x[1])
        sorted_head = sorted(self.head_char_count.items(), key=lambda x: -x[1])
        
        # 计算平均分支因子
        avg_branch = sum(len(v) for v in self.head_index.values()) / len(self.head_index) if self.head_index else 0
        
        return {
            'total_idioms': self.total_count,
            'unique_head_chars': len(self.head_index),
            'unique_tail_chars': len(self.tail_index),
            'avg_branch_factor': avg_branch,
            'top_tail_chars': sorted_tail[:10],
            'top_head_chars': sorted_head[:10],
            'max_tail_count': sorted_tail[0][1] if sorted_tail else 0,
            'min_tail_count': sorted_tail[-1][1] if sorted_tail else 0,
        }
    
    def __len__(self) -> int:
        return self.total_count
    
    def __contains__(self, idiom_id: int) -> bool:
        return idiom_id in self.idioms


def create_sample_data(n: int = 100) -> List[Tuple[int, str, Optional[str]]]:
    """
    创建示例成语数据（用于测试）
    
    Args:
        n: 生成的成语数量
    
    Returns:
        [(id, text, pinyin), ...]
    """
    # 精选示例成语，确保有良好的接龙链
    sample_idioms = [
        (0, "一心一意", "yi xin yi yi"),
        (1, "意气风发", "yi qi feng fa"),
        (2, "发扬光大", "fa yang guang da"),
        (3, "大同小异", "da tong xiao yi"),
        (4, "异口同声", "yi kou tong sheng"),
        (5, "声名远扬", "sheng ming yuan yang"),
        (6, "扬眉吐气", "yang mei tu qi"),
        (7, "气壮山河", "qi zhuang shan he"),
        (8, "河清海晏", "he qing hai yan"),
        (9, "晏安鸩毒", "yan an zhen du"),
        (10, "毒蛇猛兽", "du she meng shou"),
        (11, "兽聚鸟散", "shou ju niao san"),
        (12, "散兵游勇", "san bing you yong"),
        (13, "勇往直前", "yong wang zhi qian"),
        (14, "前功尽弃", "qian gong jin qi"),
        (15, "弃暗投明", "qi an tou ming"),
        (16, "明察秋毫", "ming cha qiu hao"),
        (17, "毫发不爽", "hao fa bu shuang"),
        (18, "爽心悦目", "shuang xin yue mu"),
        (19, "目瞪口呆", "mu deng kou dai"),
        (20, "呆若木鸡", "dai ruo mu ji"),
        (21, "鸡飞狗跳", "ji fei gou tiao"),
        (22, "跳梁小丑", "tiao liang xiao chou"),
        (23, "丑态百出", "chou tai bai chu"),
        (24, "出人头地", "chu ren tou di"),
        (25, "地大物博", "di da wu bo"),
        (26, "博学多才", "bo xue duo cai"),
        (27, "才高八斗", "cai gao ba dou"),
        (28, "斗转星移", "dou zhuan xing yi"),
        (29, "移花接木", "yi hua jie mu"),
        (30, "木已成舟", "mu yi cheng zhou"),
        (31, "舟车劳顿", "zhou che lao dun"),
        (32, "顿开茅塞", "dun kai mao se"),
        (33, "塞翁失马", "se weng shi ma"),
        (34, "马到成功", "ma dao cheng gong"),
        (35, "功成名就", "gong cheng ming jiu"),
        (36, "就事论事", "jiu shi lun shi"),
        (37, "事半功倍", "shi ban gong bei"),
        (38, "倍道而行", "bei dao er hang"),
        (39, "行云流水", "hang yun liu shui"),
        (40, "水落石出", "shui luo shi chu"),
        (41, "出神入化", "chu shen ru hua"),
        (42, "化险为夷", "hua xian wei yi"),
        (43, "夷然自若", "yi ran zi ruo"),
        (44, "若有所思", "ruo you suo si"),
        (45, "思前想后", "si qian xiang hou"),
        (46, "后继有人", "hou ji you ren"),
        (47, "人山人海", "ren shan ren hai"),
        (48, "海阔天空", "hai kuo tian kong"),
        (49, "空空如也", "kong kong ru ye"),
        (50, "也里可温", "ye li ke wen"),  # 较少见，用于测试边界
        # 添加更多高频字成语
        (51, "一马当先", "yi ma dang xian"),
        (52, "先人后己", "xian ren hou ji"),
        (53, "己所不欲", "ji suo bu yu"),
        (54, "欲速不达", "yu su bu da"),
        (55, "达官贵人", "da guan gui ren"),
        (56, "人定胜天", "ren ding sheng tian"),
        (57, "天经地义", "tian jing di yi"),
        (58, "义不容辞", "yi bu rong ci"),
        (59, "辞旧迎新", "ci jiu ying xin"),
        (60, "新陈代谢", "xin chen dai xie"),
        (61, "谢天谢地", "xie tian xie di"),
        (62, "地久天长", "di jiu tian chang"),
        (63, "长驱直入", "chang qu zhi ru"),
        (64, "入木三分", "ru mu san fen"),
        (65, "分秒必争", "fen miao bi zheng"),
        (66, "争分夺秒", "zheng fen duo miao"),
        (67, "秒杀全场", "miao sha quan chang"),  # 现代成语
        (68, "场面宏大", "chang mian hong da"),
        (69, "大张旗鼓", "da zhang qi gu"),
        (70, "鼓足干劲", "gu zu gan jin"),
        (71, "劲头十足", "jin tou shi zu"),
        (72, "足智多谋", "zu zhi duo mou"),
        (73, "谋事在人", "mou shi zai ren"),
        (74, "人才辈出", "ren cai bei chu"),
        (75, "出类拔萃", "chu lei ba cui"),
        (76, "萃于一堂", "cui yu yi tang"),
        (77, "堂堂正正", "tang tang zheng zheng"),
        (78, "正大光明", "zheng da guang ming"),
        (79, "明辨是非", "ming bian shi fei"),
        (80, "非同小可", "fei tong xiao ke"),
        (81, "可歌可泣", "ke ge ke qi"),
        (82, "泣不成声", "qi bu cheng sheng"),
        (83, "声势浩大", "sheng shi hao da"),
        (84, "大快人心", "da kuai ren xin"),
        (85, "心旷神怡", "xin kuang shen yi"),
        (86, "怡然自得", "yi ran zi de"),
        (87, "得心应手", "de xin ying shou"),
        (88, "手到擒来", "shou dao qin lai"),
        (89, "来日方长", "lai ri fang chang"),
        (90, "长话短说", "chang hua duan shuo"),
        (91, "说一不二", "shuo yi bu er"),
        (92, "二三其德", "er san qi de"),
        (93, "德才兼备", "de cai jian bei"),
        (94, "备而不用", "bei er bu yong"),
        (95, "用兵如神", "yong bing ru shen"),
        (96, "神采奕奕", "shen cai yi yi"),
        (97, "奕奕生辉", "yi yi sheng hui"),
        (98, "辉煌灿烂", "hui huang can lan"),
        (99, "烂醉如泥", "lan zui ru ni"),
    ]
    
    return sample_idioms[:n]


if __name__ == "__main__":
    # 测试数据加载
    dict_obj = IdiomDictionary(use_pinyin=True)
    dict_obj.load_from_list(create_sample_data(50))
    
    print(f"加载成语数量: {len(dict_obj)}")
    print(f"统计信息: {dict_obj.get_stats()}")
    
    # 测试接龙查询
    print(f"\n'一心一意' 可接龙的成语:")
    for id_ in dict_obj.get_followers(0):
        print(f"  - {dict_obj.get_text(id_)}")