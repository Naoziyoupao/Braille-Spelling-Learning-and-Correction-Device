from vad_thread import RealTimeVADThread
import vision.braille_inference_picture as vision
import vision.test_main as vision_test
import time
import pypinyin
import pygame
import os
from brallie_covert import BrailleConverter
import dictionary_lookup as dic
from audio_synthesis import synthesize_dots_audio

class StateMachine:
    STATE_BOOT = '开机'
    STATE_MODE1 = '模式一'
    STATE_MODE2 = '模式二'
    STATE_MODE2_WRITING = '模式二_等待书写' # 新增状态

    def __init__(self):
        # 初始化pygame mixer
        pygame.mixer.init()
        
        self.state = self.STATE_BOOT
        self.braille_converter = BrailleConverter()  # 初始化盲文转换器
        self.mode2_target_word = None
        self.mode2_target_dots = None
        self.vad_thread = RealTimeVADThread(
            aggressiveness=1,
            rate=16000,
            channels=1,
            frame_duration=30,
            min_speech_duration=0.3,
            max_speech_duration=8.0,
            silence_duration=0.5,
        )
        
        # 播放启动音频
        self.play_audio("/home/raspbarrypi/Desktop/dachuang/大创音频/您好，欢迎进入本系统.mp3")
        self.play_audio("/home/raspbarrypi/Desktop/dachuang/大创音频/请选择模式一，或者模式二.mp3")
        
        self.vad_thread.start()
        print(f"系统启动，当前状态: {self.state}")

    def play_audio(self, audio_file):
        """播放音频文件"""
        try:
            if os.path.exists(audio_file):
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                # 等待音频播放完成
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            else:
                print(f"音频文件不存在: {audio_file}")
        except Exception as e:
            print(f"播放音频时出错: {e}")

    def run(self):
        try:
            while True:
                # 状态机主循环
                if self.state == self.STATE_MODE2_WRITING:
                    # 在此状态下，不处理语音输入，直接执行视觉识别流程
                    print("进入自动拍照流程，暂停语音识别...")
                    time.sleep(3)
                    
                    # 调用视觉模块进行识别
                    recognized_braille_sets = vision_test.capture_and_recognize_braille(show_image=False)
                    
                    # 假设只处理第一行识别结果
                    if recognized_braille_sets:
                        recognized_dots_list = recognized_braille_sets[0]
                        self.compare_braille(self.mode2_target_dots, recognized_dots_list)
                    else:
                        print("视觉识别失败或未识别到任何内容。")

                    
                    # 流程结束后，返回模式二的初始状态，等待下一个词语
                    print("纠错流程结束，返回模式二。")
                    self.state = self.STATE_MODE2
                    self.play_audio("/home/raspbarrypi/Desktop/dachuang/大创音频/您要检查的词语是什么.mp3")
                    continue # 继续下一次循环，避免处理旧的语音结果

                # 正常处理语音输入
                results = self.vad_thread.get_results()
                for result in results:
                    print(f"识别结果: {result}")
                    self.handle_result(result)
                
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("正在停止...")
            self.vad_thread.clear_all_buffers()
            self.vad_thread.stop()
            pygame.mixer.quit()

    def match_state(self, result, target):
        result = result.strip()
        target = target.strip()

        print(f"\n--- 匹配检查 ---")
        print(f"输入: '{result}', 目标: '{target}'")

        # 数字匹配优先
        if ('一' in result or '1' in result) and target == '模式一':
            return True
        if ('二' in result or '2' in result) and target == '模式二':
            return True
        if (('二' in result or '2' in result) and target == '模式一') or (('一' in result or '1' in result) and target == '模式二'):
            return False
        # 整体包含优先
        if target in result:
            print(f"结果: 整体包含匹配成功")
            return True
            
        # 拼音整体包含优先
        import pypinyin
        result_py_list = [p[0] for p in pypinyin.pinyin(result, style=pypinyin.NORMAL)]
        target_py_list = [p[0] for p in pypinyin.pinyin(target, style=pypinyin.NORMAL)]
        result_py = ''.join(result_py_list)
        target_py = ''.join(target_py_list)
        print(f"输入拼音: {result_py_list}, 目标拼音: {target_py_list}")

        if target_py in result_py:
            print(f"结果: 拼音整体包含匹配成功")
            return True

        # 字符重合度
        overlap = sum(1 for c in target if c in result)
        overlap_threshold = max(2, len(target) // 2)
        print(f"字符重合度: {overlap}, 阈值: {overlap_threshold}")
        if overlap >= overlap_threshold:
            print(f"结果: 字符重合度匹配成功")
            return True

        # 拼音重合度
        py_overlap = sum(1 for c in target_py_list if c in result_py_list)
        py_overlap_threshold = max(2, len(target_py_list) // 2)
        print(f"拼音重合度: {py_overlap}, 阈值: {py_overlap_threshold}")
        if py_overlap >= py_overlap_threshold:
            print(f"结果: 拼音重合度匹配成功")
            return True
            
        print(f"结果: 匹配失败")
        print(f"----------------\n")
        return False

    def handle_result(self, result):
        if self.state == self.STATE_BOOT:
            if self.match_state(result, '模式一'):
                self.state = self.STATE_MODE1
                print("切换到模式一状态")
                self.play_audio("/home/raspbarrypi/Desktop/dachuang/大创音频/成功进入词典查询模式.mp3")
                self.play_audio("/home/raspbarrypi/Desktop/dachuang/大创音频/请说出您要查询的词语.mp3")
            elif self.match_state(result, '模式二'):
                self.state = self.STATE_MODE2
                print("切换到模式二状态")
                self.play_audio("/home/raspbarrypi/Desktop/dachuang/大创音频/成功进入书写纠错状态.mp3")
                self.play_audio("/home/raspbarrypi/Desktop/dachuang/大创音频/您要检查的词语是什么.mp3")
        
        elif self.state == self.STATE_MODE1:
            # 优先检查是否要返回
            if self.match_state(result, '返回初始状态'):
                self.state = self.STATE_BOOT
                print("已返回初始状态")
                self.play_audio("/home/raspbarrypi/Desktop/dachuang/大创音频/成功返回初始状态.mp3")
                return

            # 否则，处理为查词
            print(f"[模式一] 接收到词语: {result}")
            
            # 1. 查询字典
            definition = dic.lookup(result)
            print(f"查词结果: {definition}")
            
            # 2. 转换为盲文
            braille_info = self.braille_converter.get_braille_info(result)
            print(f"盲文转换结果:\n{braille_info}")
            
            # 3. 提取结构化的盲文点位列表
            dots_sets = self.braille_converter.get_braille_sets_for_word(result)
            print(f"结构化盲文点位列表: {dots_sets}")
            
            # 4. 合成并播放盲文点位音频
            output_audio_path = "braille_dots_audio.mp3"
            if synthesize_dots_audio(dots_sets, output_audio_path):
                # 播放提示音和合成的点位音频
                self.play_audio("/home/raspbarrypi/Desktop/dachuang/大创音频/您要查询的词语拼写方式是.mp3")
                
                self.play_audio(output_audio_path)
            else:
                print("无法合成盲文点位音频。")
            
        elif self.state == self.STATE_MODE2:
            # 优先检查是否要返回
            if self.match_state(result, '返回初始状态'):
                self.state = self.STATE_BOOT
                print("已返回初始状态")
                self.play_audio("/home/raspbarrypi/Desktop/dachuang/大创音频/成功返回初始状态.mp3")
                return

            # 检查是否是“检查”指令 - 这部分逻辑将被新的工作流替代
            if '检查' in result:
                # 提醒用户新流程
                self.play_audio("/home/raspbarrypi/Desktop/dachuang/大创音频/您要检查的词语是什么.mp3")
                return

            # 如果不是“检查”，则认为是设置目标词语
            self.mode2_target_word = result
            # 使用新函数，获取结构化的盲文点位集合列表
            self.mode2_target_dots = self.braille_converter.get_braille_sets_for_word(result)
            
            print(f"[模式二] 目标词语已设置为: '{self.mode2_target_word}'")
            print(f"[模式二] 对应的盲文点位序列为: {self.mode2_target_dots}")
            
            # 直接提示用户放置书写内容，并切换到等待状态
            self.play_audio("/home/raspbarrypi/Desktop/dachuang/大创音频/请将书写内容放在机器前方.mp3")
            self.state = self.STATE_MODE2_WRITING # 切换到新状态，主循环将接管

    def compare_braille(self, target_dots_sets, recognized_dots_sets):
        """
        比较目标盲文和识别出的盲文。
        target_dots_sets: 目标点位集合列表，例如 [{1,2}, {3,4}]
        recognized_dots_sets: 识别出的点位集合列表，例如 [{1,2}, {3,5}]
        """
        print(f"比对开始: 目标 {target_dots_sets} vs 识别 {recognized_dots_sets}")

        if target_dots_sets == recognized_dots_sets:
            print("书写完全正确！")
            self.play_audio("/home/raspbarrypi/Desktop/dachuang/大创音频/您的拼写非常正确.mp3")
        else:
            print("书写存在错误。")
            self.play_audio("/home/raspbarrypi/Desktop/dachuang/大创音频/您的拼写错误啦.mp3")
            
            # 获取整个词语的结构化点位列表用于播报
            correct_dots_sets = self.braille_converter.get_braille_sets_for_word(self.mode2_target_word)
            print(f"正确的完整点位序列是: {correct_dots_sets}")
            
            self.play_audio("/home/raspbarrypi/Desktop/dachuang/大创音频/正确的书写方式为.mp3")
            
            # 合成并播报整个词语的正确点位
            output_audio_path = "correct_word_dots.mp3"
            if synthesize_dots_audio(correct_dots_sets, output_audio_path):
                self.play_audio(output_audio_path)
            
            self.play_audio("/home/raspbarrypi/Desktop/dachuang/大创音频/请继续努力.mp3")


if __name__ == "__main__":
    sm = StateMachine()
    sm.run()