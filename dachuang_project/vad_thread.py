import threading
import pyaudio
import webrtcvad
import queue
import numpy as np
import torch
import whisper
import time
from collections import deque
import re

class RealTimeVADThread:
    def __init__(self, aggressiveness=2, rate=16000, channels=1, frame_duration=30, 
                 min_speech_duration=0.5, max_speech_duration=8.0, 
                 silence_duration=0.5, energy_threshold=10000
        ):
        
        self.vad = webrtcvad.Vad(aggressiveness)
        self.rate = rate
        self.channels = channels
        self.frame_duration = frame_duration
        self.frame_size = int(rate * frame_duration / 1000)
        
        # 语音段参数
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration
        
        # 状态和阈值参数
        self.energy_threshold = energy_threshold  # 固定能量阈值
        self.speech_to_silence_frames = int(silence_duration * 1000 / frame_duration)
        self.silence_to_speech_frames = 3  # 连续3帧语音才触发

        # 状态管理
        self.state = 'SILENT'
        self.speech_buffer = deque()
        self.pre_buffer = deque(maxlen=self.silence_to_speech_frames) # 预录制缓冲区
        self.consecutive_speech_count = 0
        self.consecutive_silence_count = 0
        
        # Whisper模型配置 - 修正参数
        print("正在加载Whisper模型...")
        self.model = whisper.load_model("tiny")
        
        # 正确的Whisper转录参数
        self.transcribe_options = {
            'language': 'zh',  # 简体中文
            'task': 'transcribe',
            'fp16': False,
            'no_speech_threshold': 0.6,
            'logprob_threshold': -1.0,
            'compression_ratio_threshold': 2.4,
            'condition_on_previous_text': True,
            'temperature': 0.0,
            'initial_prompt' :"以下是普通话转写为简体中文的内容：",
        }
        print("Whisper模型加载完成")
        
        # 音频流
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=channels,
                                  rate=rate,
                                  input=True,
                                  frames_per_buffer=self.frame_size)
        
        # 线程控制
        self.running = False
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        
        # 结果队列
        self.result_queue = queue.Queue()

    def start(self):
        self.running = True
        self.thread.start()
        print("VAD后台线程已启动...")

    def stop(self):
        self.running = False
        self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print("VAD后台线程已停止。")

    def _is_valid_audio(self, audio_np):
        """检查音频是否有效，避免处理噪音"""
        energy = np.mean(audio_np ** 2)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_np)))) / len(audio_np)
        min_energy = 1e-6
        max_zcr = 0.3
        return energy > min_energy and zero_crossings < max_zcr

    def _post_process_text(self, text):
        """后处理识别文本，过滤基于重复率的幻觉。"""
        if not text:
            return ""
        
        text = text.strip()
        n = len(text)
        
        # 规则1: 文本太短，不处理
        if n < 5:
            return text
            
        # 规则2: 计算最常见字符的频率
        from collections import Counter
        most_common = Counter(text).most_common(1)
        if not most_common:
            return text
            
        char, count = most_common[0]
        
        # 如果最常见字符的占比超过80%，则认为是幻觉
        repetition_ratio = count / n
        if repetition_ratio > 0.8:
            print(f"检测到高频重复幻觉 (字符 '{char}' 占比 {repetition_ratio:.2%})，已过滤: '{text}'")
            return ""
            
        return text

    def _run(self):
        while self.running:
            try:
                audio = self.stream.read(self.frame_size, exception_on_overflow=False)
                
                is_speech_by_vad = self.vad.is_speech(audio, self.rate)
                audio_data = np.frombuffer(audio, dtype=np.int16)
                energy = np.sqrt(np.mean(audio_data.astype(float)**2))
                
                # 使用固定的能量阈值
                is_speech = is_speech_by_vad and (energy > self.energy_threshold)

                if self.state == 'SILENT':
                    # 预录制缓冲区持续记录最近的音频帧
                    self.pre_buffer.append(audio)

                    if is_speech:
                        self.consecutive_speech_count += 1
                        if self.consecutive_speech_count >= self.silence_to_speech_frames:
                            print(f"检测到语音开始 (能量: {energy:.0f} > 阈值: {self.energy_threshold})")
                            self.state = 'SPEECH'
                            # 将预录制缓冲区的内容转存到主缓冲区
                            self.speech_buffer.clear()
                            self.speech_buffer.extend(self.pre_buffer)
                            self.consecutive_silence_count = 0
                    else:
                        # 如果不是语音，重置连续语音计数
                        self.consecutive_speech_count = 0

                elif self.state == 'SPEECH':
                    if is_speech:
                        self.speech_buffer.append(audio)
                        self.consecutive_silence_count = 0
                        
                        # 检查是否达到最大时长
                        if len(self.speech_buffer) * self.frame_duration / 1000.0 >= self.max_speech_duration:
                            print(f"达到最大语音持续时间({self.max_speech_duration:.2f}秒)，开始识别...")
                            self._process_speech_segment()
                            self.state = 'SILENT'
                            self.consecutive_speech_count = 0
                    else:
                        self.consecutive_silence_count += 1
                        if self.consecutive_silence_count >= self.speech_to_silence_frames:
                            duration = len(self.speech_buffer) * self.frame_duration / 1000.0
                            if duration >= self.min_speech_duration:
                                print(f"检测到语音结束，持续约{duration:.2f}秒，开始识别...")
                                self._process_speech_segment()
                            else:
                                print(f"语音段太短({duration:.2f}秒)，忽略")
                            
                            self.state = 'SILENT'
                            self.consecutive_speech_count = 0
                            
            except Exception as e:
                print(f"处理音频时出错: {e}")
                self.state = 'SILENT'
                self.speech_buffer.clear()
                self.consecutive_speech_count = 0
                self.consecutive_silence_count = 0

    def _process_speech_segment(self):
        """处理收集到的语音段"""
        if not self.speech_buffer:
            return
            
        try:
            audio_data = b''.join(self.speech_buffer)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            audio_duration = len(audio_np) / self.rate
            print(f"正在识别语音段，长度: {audio_duration:.2f}秒")
            
            if not self._is_valid_audio(audio_np):
                print("音频能量过低，可能是噪音，跳过识别")
                return
            
            # 使用正确的参数调用transcribe
            result = self.model.transcribe(
                audio_np,
                **self.transcribe_options
            )
            
            text = result["text"].strip()
            print(f"原始识别结果: {text}")
            
            # 后处理文本
            processed_text = self._post_process_text(text)
            
            if processed_text:
                print(f"最终结果: {processed_text}")
                self.result_queue.put(processed_text)
            else:
                print("未识别到有效文本或文本被过滤")
                
        except Exception as e:
            print(f"语音识别出错: {e}")
        
        finally:
            self.speech_buffer.clear()

    def get_results(self):
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def clear_all_buffers(self):
        self.speech_buffer.clear()
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

if __name__ == "__main__":
    vad_thread = RealTimeVADThread(
        aggressiveness=2,
        rate=16000, 
        channels=1, 
        frame_duration=30,
        min_speech_duration=0.5,
        max_speech_duration=8.0,
        silence_duration=0.8, # 判定结束的静音时长
        energy_threshold=5000 # 固定能量阈值
    )
    
    vad_thread.start()
    
    try:
        while True:
            results = vad_thread.get_results()
            for result in results:
                print(f"主线程收到结果: {result}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n正在停止...")
        vad_thread.clear_all_buffers()
        vad_thread.stop()