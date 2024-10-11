import time
import numpy as np

import queue
from queue import Queue
import multiprocessing as mp


class BaseASR:
    def __init__(self, opt, parent=None):
        self.opt = opt
        self.parent = parent

        self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000) 每一帧包含的音频样本数 320
        self.queue = Queue() ## 存储输入的音频帧
        self.output_queue = mp.Queue() ## 存储输出的音频帧，多进程队列，用于不同进程之间传递音频数据

        self.batch_size = opt.batch_size

        self.frames = [] ## 缓存音频帧，用于后面特征提取中从多个帧中生成音频特征
        self.stride_left_size = opt.l
        self.stride_right_size = opt.r ## 滑动窗口的大小，用于控制特征提取时需要的历史和未来帧数
        #self.context_size = 10
        self.feat_queue = mp.Queue(2) ## 多进程队列，用于存储提取后的音频特征

        #self.warm_up()

    def pause_talk(self): ## 清空音频帧队列，目的是暂停音频处理，暂停说话
        """
        self.queue 清空
        """

        self.queue.queue.clear() 

    def put_audio_frame(self,audio_chunk): #16khz 20ms pcm 将PCM格式的16kHz采样率音频放入音频帧队列中，每个帧包含320个样本
        """
        self.queue 新增内容
        """
        self.queue.put(audio_chunk)

    def get_audio_frame(self): ## 从队列中获取音频帧，若队列为空则返回一个填充0的空帧，若有父类则需要从父类中获取音频流数据
        """
        type 0: 非静音音频
        type 1: 静音音频
        type >1: 自定义音频
        self.queue 丢掉东西
        """
        try:
            frame = self.queue.get(block=True,timeout=0.01)
            type = 0
            #print(f'[INFO] get frame {frame.shape}')
        except queue.Empty:
            if self.parent and self.parent.curr_state>1: #播放自定义音频
                frame = self.parent.get_audio_stream(self.parent.curr_state)
                type = self.parent.curr_state
            else:
                frame = np.zeros(self.chunk, dtype=np.float32)
                type = 1

        return frame,type 

    def is_audio_frame_empty(self)->bool: ## 检查音频帧队列是否为空
        return self.queue.empty()

    def get_audio_out(self):  #get origin audio pcm to nerf
        """
        self.output_queue 丢掉东西
        """
        return self.output_queue.get()
    
    def warm_up(self): ## 系统启动时预热音频队列，从输入音频帧队列queue中获取一批音频帧，并存入frames列表中（缓存），同时将这些帧传入output_queue中，为系统提供初始数据
        """
        self.queue 丢掉东西
        self.frames 新增内容
        self.output_queue 新增内容

        流向： self.queue -> self.frames, self.output_queue, self.output_queue再丢掉
        """
        for _ in range(self.stride_left_size + self.stride_right_size):
            audio_frame,type=self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame,type))
        for _ in range(self.stride_left_size):
            self.output_queue.get()

    def run_step(self):
        pass

    def get_next_feat(self,block,timeout): ## 从特征队列中获取下一个特征，用于模型的输入
        return self.feat_queue.get(block,timeout)