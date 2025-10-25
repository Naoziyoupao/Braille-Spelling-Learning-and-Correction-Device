from pydub import AudioSegment
import os

# 音频文件所在的目录
AUDIO_DIR = "/home/raspbarrypi/Desktop/dachuang/大创音频/"

# 定义点数对应的音频文件名
DOT_AUDIO_FILES = {
    1: "点1.mp3",
    2: "点2.mp3",
    3: "点3.mp3",
    4: "点4.mp3",
    5: "点5.mp3",
    6: "点6.mp3",

}

def synthesize_dots_audio(dots_sets_list, output_path):
    """
    根据点位集合的列表，将对应的音频拼接成一个完整的音频文件。

    :param dots_sets_list: 包含点位集合的列表，例如 [{1, 3, 4, 5}, {1, 2, 5}]
    :param output_path: 合成后音频的保存路径，例如 "output.mp3"
    :return: 如果成功，返回True；否则返回False。
    """
    print(f"开始合成点位音频，目标路径: {output_path}")
    
    # 创建一个空的音频段，作为拼接的起点
    combined_audio = AudioSegment.empty()
    
    try:
        # 遍历每个盲文字符的点位集合
        for dot_set in dots_sets_list:
            # 对每个集合内的点位进行排序，以保证播报顺序
            for dot in sorted(list(dot_set)):
                if dot in DOT_AUDIO_FILES:
                    audio_file_path = os.path.join(AUDIO_DIR, DOT_AUDIO_FILES[dot])
                    if os.path.exists(audio_file_path):
                        # 加载点数对应的音频文件
                        dot_audio = AudioSegment.from_mp3(audio_file_path)
                        # 拼接到主音频上
                        combined_audio += dot_audio
                    else:
                        print(f"警告: 音频文件不存在: {audio_file_path}")
                else:
                    print(f"警告: 未知的点数: {dot}，已忽略。")

        if len(combined_audio) > 0:
            # 导出拼接好的音频文件
            combined_audio.export(output_path, format="mp3")
            print(f"音频合成成功，已保存至: {output_path}")
            return True
        else:
            print("没有可合成的音频内容。")
            return False
            
    except Exception as e:
        print(f"音频合成时发生错误: {e}")
        return False

if __name__ == '__main__':
    # --- 使用示例 ---
    # 假设这是从 get_braille_sets_for_word 方法得到的点位列表
    # "你好" -> 你(简写), h, ao
    test_dots_sets = [{1, 3, 4, 5}, {1, 2, 5}, {2, 3, 5}] 
    output_file = "braille_dots_audio.mp3"
    
    # 调用函数合成音频
    success = synthesize_dots_audio(test_dots_sets, output_file)
    
    if success:
        print(f"测试音频已生成: {output_file}")
        # 你可以在这里添加代码来播放这个文件进行测试
        # 例如，使用 pygame
        try:
            import pygame
            import time
            pygame.mixer.init()
            pygame.mixer.music.load(output_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.quit()
            print("测试音频播放完毕。")
        except ImportError:
            print("未找到pygame，无法自动播放测试音频。")
        except Exception as e:
            print(f"播放测试音频时出错: {e}")
